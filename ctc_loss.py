import numpy as np
import timeit
import operator
from logger import log


def merge_two_dicts(x1, y1):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x1.copy()
    z.update(y1)
    return z


def remove_entries(y2, x2):
    """Remove entries from dictionary for given keys in the list"""
    for key in x2:
        if key in y2:
            del y2[key]
    return y2


def calculate_ctc_loss(y, alphabet="-' abcdefghijklmnopqrstuvwxyz", w=10, e="expected output"):
    """
    :param y: log probabilities of outputs(dimensions: Time * labels)
    :param alphabet: default is initialized- label 1 is treated as blank
    :param w: Beam size
    :param e: Expected output (string)
    :return: most probable output, CTC Loss
    """
    start = timeit.default_timer()
    blank = alphabet[0]
    time_step = y.shape[0]
    final_output = {}
    for aa, xx in np.ndenumerate(y[0]):
        c = alphabet[aa[0]]
        final_output[c] = xx

    for t in range(1, time_step):
        # print t
        dict_w = {}
        # k most probable sequences in final_output
        if any(final_output):
            sorted_dict = sorted(final_output.items(), key=operator.itemgetter(1), reverse=True)
            for item in sorted_dict[0:w]:
                dict_w[item[0]] = item[1]

        temp_add = {}
        temp_remove = []
        for index, x in np.ndenumerate(y[t]):
            c = alphabet[index[0]]
            # if x < -20.20:
                # continue
            if any(dict_w):
                for key, value in dict_w.iteritems():
                    # Check for last character of the values in dictionary key
                    # If equal, make the entry in temp_add with same key with...
                    # Multiply the probabilities, i.e. add the log of probabilities
                    if key[-1] == c:
                        if key not in temp_remove:
                            temp_remove.append(key)
                        if key not in temp_add:
                            temp_add[key] = value+x
                        else:
                            dd = temp_add[key]
                            temp_add[key] = np.logaddexp(value+x, dd)
                    elif key[-1] == blank:
                        # If different, either remove the previous blank from the key and
                        # add new item with new key (ab instead of a_b)
                        # or simply add in temp_add with key+new_character
                        if key not in temp_remove:
                            temp_remove.append(key)
                        if key not in temp_add:
                            temp_add[key[:-1] + c] = value+x
                        else:
                            dd = temp_add[key]
                            temp_add[key[:-1] + c] = np.logaddexp(value+x, dd)
                    else:
                        temp_add[key + c] = value + x

        # Merge two dictionaries
        # Before merging update the values in temp_add as sum of probabilities for the existing entries
        if any(final_output):
            for k, v in temp_add.iteritems():
                if k in final_output:
                    temp_add[k] = np.logaddexp(temp_add[k], temp_add[k])

        # Remove entries from dictionary
        if temp_remove:
            final_output = remove_entries(final_output, temp_remove)

        final_output = merge_two_dicts(final_output, temp_add)

    stop = timeit.default_timer()
    log.info("ctc_loss running time : " + str(stop-start))

    # Calculate loss and most probable output
    most_probable_output = max(final_output.iteritems(), key=operator.itemgetter(1))[0]
    if e in final_output:
        ctc_loss = final_output[e] / sum(final_output.values())
    else:
        ctc_loss = 1.0
    log.info("Most probable output : " + str(most_probable_output))
    log.info("CTC loss: " + str(ctc_loss))
    return most_probable_output, ctc_loss


if __name__ == '__main__':
    # Sample call
    dist = np.log(np.array([[0.02, 0.03, 0.05],
                            [0.04, 0.04, 0.02],
                            [0.01, 0.03, 0.06],
                            [0.05, 0.02, 0.03]]))
    print dist
    eee = np.loadtxt("./temp_temp.txt")
    o = calculate_ctc_loss(eee, e='yz')
    print o
