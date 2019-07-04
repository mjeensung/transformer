import unittest
from tokenizer import WordpieceTokenizer
class TddTest(unittest.TestCase):
    def testTokenizer(self):
        tokenizer = WordpieceTokenizer()
        x = tokenizer.transform(["I loved his earring"])
        print(x)
        # 결과 값이 일치 여부 확인
        # self.assertEqual(x, [""])

if __name__ == '__main__':
    unittest.main()