import argparse


def substring_between(text: str, beg: str, end: str) -> str:
    return text.split(beg)[1].split(end)[0]


def split_and_strip(text: str, sep: str) -> list[str]:
    return list(map(lambda s: s.strip(), text.split(sep)))


def handle_alignment(alignment: str) -> str:
    data = split_and_strip(alignment, '//')
    cls = data[1].split('_')[0]
    score = '0' if data[2] == 'NIL' else data[2]
    chunks = list(map(lambda c: '' if c == '-not aligned-' else c, split_and_strip(data[3], '<==>')))
    result = [chunks[0], chunks[1], cls, score]
    return '\t'.join(result)


def handle_sentence(sentence: str) -> str:
    alignments = filter(lambda s: s != '',
                        split_and_strip(substring_between(sentence, '<alignment>', '</alignment>'), '\n'))
    return '\n'.join(map(handle_alignment, alignments))


def handle_wa(wa_content: str) -> str:
    sentences = filter(lambda s: s != '', map(lambda s: s.strip(), wa_content.split('<sentence')))
    return '\n'.join(map(handle_sentence, sentences))


def main():
    parser = argparse.ArgumentParser(description='.wa to .tsv converter')
    parser.add_argument('file_path')

    args = parser.parse_args()

    with open(args.file_path) as wa:
        wa_content = wa.read()

    tsv_content = '\t'.join(['x1', 'x2', 'y_type', 'y_score']) + '\n' + handle_wa(wa_content)

    output_path = f'{"/".join(args.file_path.split("/")[:-1])}/{"test" if "test" in args.file_path else "train"}.tsv'

    with open(output_path, 'w') as tsv:
        tsv.write(tsv_content)


if __name__ == '__main__':
    main()
