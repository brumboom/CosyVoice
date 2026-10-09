"""Frontend regression tests; no model weights or normalizer downloads needed."""
import unittest
import json
from unittest.mock import Mock

import inflect

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.frontend_utils import contains_cyrillic, split_cyrillic_text


class CyrillicFrontendTest(unittest.TestCase):
    def setUp(self):
        self.frontend = CosyVoiceFrontEnd.__new__(CosyVoiceFrontEnd)
        self.frontend.tokenizer = Mock()
        self.frontend.tokenizer.encode.side_effect = lambda text, **kwargs: list(text)
        self.frontend.allowed_special = 'all'
        self.frontend.inflect_parser = inflect.engine()
        self.frontend.text_frontend = ''
        self.frontend.en_tn_model = Mock()
        self.frontend.zh_tn_model = Mock()
        self.frontend.frd = Mock()
        self.frontend.en_tn_model.normalize.side_effect = lambda text: text
        self.frontend.zh_tn_model.normalize.side_effect = lambda text: text
        self.frontend.frd.do_voicegen_frd.side_effect = lambda text: json.dumps({'sentences': [{'text': text}]})

    def test_cyrillic_preserved_for_each_backend(self):
        samples = [
            'У неё 12 яблок и 3 груши.',
            'Дата: 09.10.2026, время: 15:30, цена: 3,14 руб.',
            'Ёж шёл.\nОн нашёл жёлтый лист!',
            'Версия API v2: 中文, English и русский — вместе.',
            'МГУ, ул. Ленина, д. 12; т. е. следующий дом...',
            'Без конечной точки',
            ('Длинная русская фраза с числом 1250. ' * 12).rstrip(),
        ]
        for backend in ('', 'wetext', 'ttsfrd'):
            self.frontend.text_frontend = backend
            for text in samples:
                with self.subTest(backend=backend, text=text):
                    self.assertEqual(self.frontend.text_normalize(text, split=False), text)
                    self.assertEqual(''.join(self.frontend.text_normalize(text)), text)
        self.frontend.en_tn_model.normalize.assert_not_called()
        self.frontend.zh_tn_model.normalize.assert_not_called()
        self.frontend.frd.do_voicegen_frd.assert_not_called()

    def test_disabled_and_special_tokens_are_unchanged(self):
        text = '  Ёж 12  '
        self.assertEqual(self.frontend.text_normalize(text, text_frontend=False), [text])
        self.assertEqual(self.frontend.text_normalize(text, split=False, text_frontend=False), text)
        prompt = 'You are a helpful assistant.<|endofprompt|>У неё 12 яблок.'
        self.assertEqual(self.frontend.text_normalize(prompt, split=False), prompt)
        generator = (part for part in ['Привет', ' мир'])
        self.assertIs(self.frontend.text_normalize(generator)[0], generator)
        self.assertEqual(self.frontend.text_normalize(''), [''])
        self.assertEqual(self.frontend.text_normalize('!!!'), [])

    def test_existing_english_and_chinese_paths(self):
        self.assertEqual(self.frontend.text_normalize('There are 12 apples.', split=False), 'There are twelve apples.')
        self.assertEqual(self.frontend.text_normalize('你好。'), ['你好。'])
        self.frontend.text_frontend = 'wetext'
        self.frontend.en_tn_model.normalize.side_effect = None
        self.frontend.en_tn_model.normalize.return_value = 'There are twelve apples.'
        self.assertEqual(self.frontend.text_normalize('There are 12 apples.'), ['There are twelve apples.'])
        self.frontend.en_tn_model.normalize.assert_called_once_with('There are 12 apples.')
        self.frontend.zh_tn_model.normalize.return_value = '你好。'
        self.frontend.zh_tn_model.normalize.side_effect = None
        self.assertEqual(self.frontend.text_normalize('你好。'), ['你好。'])
        self.frontend.zh_tn_model.normalize.assert_called_once_with('你好。')
        self.frontend.text_frontend = 'ttsfrd'
        self.frontend.frd.do_voicegen_frd.side_effect = None
        self.frontend.frd.do_voicegen_frd.return_value = '{"sentences": [{"text": "Hello."}]}'
        self.assertEqual(self.frontend.text_normalize('Hello.'), ['Hello.'])
        self.frontend.frd.do_voicegen_frd.assert_called_once_with('Hello.')

    def test_other_languages_keep_existing_contract(self):
        for text in ('Hallo Welt.', 'Hola mundo.', 'Bonjour.', 'Ciao.', 'こんにちは。', '안녕하세요.'):
            with self.subTest(text=text):
                self.assertEqual(self.frontend.text_normalize(text, split=False), text)

    def test_grouping_preserves_boundaries_and_content(self):
        text = 'Ёж пришёл 09.10.2026 в 15:30. ' * 10 + 'Конец...'
        chunks = split_cyrillic_text(text, list)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(''.join(chunks), text)
        self.assertEqual(sum(chunk.count('09.10.2026') for chunk in chunks), 10)
        self.assertEqual(sum(chunk.count('15:30') for chunk in chunks), 10)
        sentence = 'Я' * 200
        self.assertEqual(split_cyrillic_text(sentence, list), [sentence])

    def test_unicode_script_detection(self):
        for text in ('Ё', 'ё', 'Я', 'Ѣ', 'Ꙁ', '中文 русский', 'API 2 русский'):
            self.assertTrue(contains_cyrillic(text))
        for text in ('Hello', '你好', 'こんにちは', '안녕하세요', 'Hallo', 'Hola', 'Bonjour', 'Ciao', '12'):
            self.assertFalse(contains_cyrillic(text))


if __name__ == '__main__':
    unittest.main()
