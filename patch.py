# 1 batch_size
batch_size = min(batch_size, len(dataset))

# 2 number of workers
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

