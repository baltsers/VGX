def abs(length):
    return length if length >= 0 else 0



def decoder_process(target, vocab, max_target_len, e_voc=None, pointer=False):
    f_s = [vocab.find(sub_token) for sub_token in target]  # f_s not should been map, because need embedded
    if not pointer:
        f_t = [vocab.find(sub_token) for sub_token in target]
    else:
        assert e_voc is not None
        f_t = []
        for sub_token in target:
            if sub_token in e_voc:
                f_t.append(e_voc[sub_token])
            else:
                f_t.append(vocab.find(sub_token))
                # here still exist unk in f_t, cause perhaps some word not presented in method body
    f_source = [vocab.sos_index] + f_s
    f_source = f_source[:max_target_len] + [vocab.pad_index] * abs(max_target_len - len(f_source))
    # f_source: max_target_len
    f_target = f_t + [vocab.eos_index]
    f_target = f_target[:max_target_len] + [vocab.pad_index] * abs(max_target_len - len(f_target))
    # f_target: max_target_len

    return f_source, f_target


def row_process(row, max_code_length):
    min_row = min(row)
    row = [num - min_row + 1 for num in row]  # 0 for padding
    row_ = row[:max_code_length] + [0] * abs(max_code_length - len(row))
    return row_


def content_process(content, named, vocab, max_code_length, e_voc=None, pointer=False):
    content_ = [vocab.find(token) for token in content]
    if not pointer:
        content_e = None
    else:
        assert e_voc is not None
        content_e = []
        for sub_token in content:
            if sub_token in e_voc:
                content_e.append(e_voc[sub_token])
            else:
                content_e.append(vocab.find(sub_token))
    content_ = content_[:max_code_length] + [vocab.pad_index] * abs(max_code_length - len(content))
    if pointer:
        content_e = content_e[:max_code_length] + [vocab.pad_index] * abs(max_code_length - len(content))
    # content_: max_code_length
    content_mask_ = [1 for _ in content][:max_code_length] + [0] * abs(
        max_code_length - len(content))
    named_ = named[:max_code_length] + [2] * abs(max_code_length - len(named))  # 2 for padding
    return content_, content_mask_, named_, content_e


def make_extended_vocabulary(content, vocab):
    e_voc, e_voc_ = dict(), dict()
    idx = len(vocab)
    for token in content:
        if not vocab.has_token(token) and token not in e_voc:
            e_voc[token] = idx
            e_voc_[idx] = token
            idx += 1
    return e_voc, e_voc_, idx
