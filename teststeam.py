from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

text = "film ini sangat menyenangkan dan menghibur sekali"
print(stemmer.stem(text))
