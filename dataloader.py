
from multiprocessing import Process, Pool
from knowledgeable import add_knowledge_worker

def read_dataset(path, workers_num=1):

    print("Loading sentences from {}".format(path))
    sentences = []
    with open(path, mode='r', encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            sentences.append(line)
    sentence_num = len(sentences)

    print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
    if workers_num > 1:
        params = []
        sentence_per_block = int(sentence_num / workers_num) + 1
        for i in range(workers_num):
            params.append((i, sentences[i*sentence_per_block: (i+1)*sentence_per_block], config.columns, kg, vocab, args))
        pool = Pool(workers_num)
        res = pool.map(add_knowledge_worker, params)
        pool.close()
        pool.join()
        dataset = [sample for block in res for sample in block]
    else:
        params = (0, sentences, config.columns, kg, vocab, args)
        dataset = add_knowledge_worker(params)

    return dataset

# Datset loader.
def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms):
    instances_num = input_ids.size()[0]
    for i in range(instances_num // batch_size):
        input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
        label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
        mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
        pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
        vms_batch = vms[i*batch_size: (i+1)*batch_size]
        yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch
    if instances_num > instances_num // batch_size * batch_size:
        input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
        label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
        mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
        pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
        vms_batch = vms[instances_num//batch_size*batch_size:]

        yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch

