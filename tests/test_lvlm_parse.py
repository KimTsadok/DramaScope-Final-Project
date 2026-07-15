import unittest

from src.lvlm.parse import extract_json_block, parse_structured_response


class StructuredResponseParsingTests(unittest.TestCase):
    def test_direct_json_is_parsed(self):
        parsed = parse_structured_response(
            '{"summary": "hello", "interaction_level": 2}'
        )

        self.assertEqual(parsed["summary"], "hello")
        self.assertEqual(parsed["interaction_level"], 2)

    def test_fenced_json_is_parsed(self):
        parsed = parse_structured_response(
            '```json\n{"summary": "hello", "interaction_level": 1}\n```'
        )

        self.assertEqual(parsed["summary"], "hello")
        self.assertEqual(parsed["interaction_level"], 1)

    def test_surrounding_commentary_is_ignored(self):
        parsed = parse_structured_response(
            'Model output: {"summary": "hello", "interaction_level": 3} done.'
        )

        self.assertEqual(parsed["summary"], "hello")
        self.assertEqual(parsed["interaction_level"], 3)

    def test_braces_inside_strings_do_not_end_or_extend_object(self):
        for summary in ("a } shaped object", "a { shaped object"):
            with self.subTest(summary=summary):
                response = (
                    "prefix "
                    + '{"summary": '
                    + repr(summary).replace("'", '"')
                    + ', "interaction_level": 2} suffix'
                )

                parsed = parse_structured_response(response)

                self.assertEqual(parsed["summary"], summary)

    def test_escaped_quotes_backslashes_and_nested_data_are_supported(self):
        response = (
            'prefix {"summary":"brace } and \\"quote\\" and \\\\ path",'
            '"interaction_level":2,"nested":{"items":[1,2]}} suffix'
        )

        json_block = extract_json_block(response)
        parsed = parse_structured_response(response)

        self.assertIn('"nested"', json_block)
        self.assertEqual(parsed["interaction_level"], 2)
        self.assertIn("brace }", parsed["summary"])

    def test_missing_or_malformed_object_is_rejected(self):
        for response in ("no object here", 'prefix {"summary": "unfinished"'):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    parse_structured_response(response)

    def test_json_array_root_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a JSON object"):
            parse_structured_response('[{"summary": "hello"}]')


if __name__ == "__main__":
    unittest.main()
