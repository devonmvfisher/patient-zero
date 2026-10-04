# SPDX-License-Identifier: MIT OR Apache-2.0
"""Synthetic contracts for published counts and documented extraction rules."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from patient_zero import cascade as C, load as L, tokens as T


def make_rows(n, text='GARDENBELL', mode='chain'):
    rows = []
    for i in range(n):
        human = mode == 'unnamed' and i == 0
        place = 'garden/' + str(i) if mode == 'parallel' else 'garden'
        if mode == 'later-seed' and i == 0:
            place = 'separate'
        rows.append(L.Row('r' + str(i), None if human else 'Graft' + str(i),
                         L.ROLE_HUMAN if human else L.ROLE_AGENT, i * 100.0, 0.0, None, place, None, None, text))
    return rows


def tree_counts(n, mode='chain'):
    rows = make_rows(n, mode=mode)
    token = T.Token('code:GARDENBELL', 'code', 'code', 'GARDENBELL', tuple(range(n)),
                    n - int(mode == 'unnamed'), (), ())
    return C.build_trees(rows, {token.key: token})[1]


class SummaryContracts(unittest.TestCase):
    def test_chain_has_one_tree_and_three_named_links(self):
        stats = tree_counts(4)
        self.assertEqual(stats['classes']['code'], {'trees': 1, 'named_links': 3, 'parallel_starts': 0,
                                                  'dropped_by_clock': 0, 'trees_with_links': 1})
        self.assertEqual(stats['trees_with_links_by_agent_count'], {'3-4': 1, '5-9': 0, '10-49': 0, '50+': 0})
        self.assertEqual(stats['trees_seed_is_not_first_user'], 0)
        self.assertEqual(stats['trees_without_effective_seed'], 0)

    def test_root_and_unnamed_counts(self):
        self.assertEqual(tree_counts(4, 'parallel')['trees_without_effective_seed'], 1)
        self.assertEqual(tree_counts(4, 'later-seed')['trees_seed_is_not_first_user'], 1)
        self.assertEqual(tree_counts(4, 'unnamed')['links_from_exposure_only_rows'], 1)

    def test_tree_agent_bucket_boundaries(self):
        for n, bucket in [(5, '5-9'), (9, '5-9'), (10, '10-49'), (11, '10-49'), (50, '50+')]:
            with self.subTest(n=n):
                counts = tree_counts(n)['trees_with_links_by_agent_count']
                self.assertEqual(counts[bucket], 1)
                self.assertEqual(sum(counts.values()), 1)

    def test_token_agent_bucket_boundaries(self):
        for n, bucket in [(5, '5-9'), (9, '5-9'), (10, '10-49'), (11, '10-49')]:
            with self.subTest(n=n):
                found, stats = T.find_tokens(make_rows(n, text='`orchard`'))
                self.assertEqual(set(found), {'code:orchard'})
                self.assertEqual(stats['kept_by_agent_count'][bucket], 1)
                self.assertEqual(sum(stats['kept_by_agent_count'].values()), 1)

    def test_fourth_agent_cannot_resolve_the_first_three(self):
        rows = [row._replace(time=0.0 if i < 3 else 100.0, uncertainty=10.0)
                for i, row in enumerate(make_rows(4, text='`orchard`'))]
        found, _ = T.find_tokens(rows)
        self.assertNotIn('code:orchard', found)

    def test_backtick_words_and_phrases_remain_distinct(self):
        self.assertIn('code:orchard', T.row_tokens('`orchard`')[0])
        self.assertFalse(any(key.startswith('code:') for key in T.row_tokens('`velvet clover lantern`')[0]))

    def test_phrase_requires_two_content_words(self):
        self.assertFalse(any(key.startswith('phrase:') for key, *_ in T.line_tokens('or amber and')[0]))

    def test_text_before_url_keeps_its_first_letter(self):
        keys = T.row_tokens('amber copper meadow https://sprout.example.test/path')[0]
        self.assertIn('phrase:amber copper meadow', keys)

    def test_repeated_token_unions_risky_types(self):
        found = T.row_tokens('garden_mark proxy\ngarden_mark share')[0]
        self.assertEqual(found['code:garden_mark'][3], frozenset(('proxy', 'share')))

    def test_short_valid_colon_address_is_masked(self):
        value = 'abc' + '::' + 'de'
        self.assertEqual(L.scrub_ips(value), ('[ip]', 1))

    def test_repetitive_wiki_body_only_adds_the_new_line(self):
        before = '\n'.join(['repeated leaf'] * 250)
        self.assertEqual(L.added_text('new fern\n' + before, [], before), 'new fern')

    def test_generic_group_counts(self):
        with tempfile.TemporaryDirectory(prefix='pz-groups-') as folder:
            path = Path(folder) / 'synthetic.ndjson'
            rows = [dict(agent='Graft' + str(i), time=i, place='garden', text='SYNTHETIC heading',
                         group='one' if i < 2 else 'two') for i in range(3)]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
            self.assertEqual(L.load_generic(str(path))[1]['groups'], {'one': 2, 'two': 1})

    def test_wiki_unknown_pages_and_optional_uncertainty(self):
        for unknown, uncertainty in [(False, 0), (True, 0), (False, None), (False, True)]:
            with self.subTest(unknown=unknown, uncertainty=uncertainty), tempfile.TemporaryDirectory(prefix='pz-wiki-counts-') as folder:
                root = Path(folder)
                revision = dict(rev_id='revision', page_id='garden', wiki='SYNTHETIC', label='Graft', time=100,
                                body='SYNTHETIC fresh leaves', hunks=[], diff_base=None, uncertainty_seconds=uncertainty)
                for name, values in [('revisions', [revision]), ('pages', [] if unknown else [dict(page_id='garden', page_family='leaf')]), ('labels', [])]:
                    with gzip.open(root / (name + '.jsonl.gz'), 'wt', encoding='utf-8') as stream:
                        stream.write(''.join(json.dumps(v) + '\n' for v in values))
                rows, stats = L.load_wiki(str(root))
                self.assertEqual(stats['saves_on_unknown_page'], int(unknown))
                self.assertEqual(rows[0].uncertainty, 0.0)
