import argparse

fields_sep = ' // '


def preds_to_wa(wa_content: str, preds_content: str):
    wa_lines = wa_content.splitlines()
    preds_lines = preds_content.splitlines()

    idx = 0
    result = []

    for line in wa_lines:
        line_res = line

        if '<==>' in line:
            fields = line.split(fields_sep)
            preds_fields = preds_lines[idx].split()

            fields[1] = preds_fields[1]
            fields[2] = preds_fields[2]

            line_res = fields_sep.join(fields)
            idx += 1

        result.append(line_res)
    return '\n'.join(result)


def main():
    parser = argparse.ArgumentParser(description='.wa to .tsv converter')
    parser.add_argument('wa_path')
    parser.add_argument('preds_path')

    args = parser.parse_args()

    with open(args.wa_path) as wa:
        wa_content = wa.read()

    with open(args.preds_path) as preds:
        preds_content = preds.read()

    wa_preds_content = preds_to_wa(wa_content, preds_content)

    output_path = f'{".".join(args.preds_path.split(".")[:-1])}.wa'

    with open(output_path, 'w') as tsv:
        tsv.write(wa_preds_content)


if __name__ == '__main__':
    main()
