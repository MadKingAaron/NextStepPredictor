from pynvml import *
import math

nvmlInit()

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 4)
   return "%s %s" % (s, size_name[i])

def get_gpu_mem(gpu:int):
    h = nvmlDeviceGetHandleByIndex(gpu)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'GPU: {gpu}')
    print(f'total    : {convert_size(info.total)}')
    print(f'free     : {convert_size(info.free)}')
    print(f'used     : {convert_size(info.used)}')


def get_best_free_gpu(gpus:int):
    best_gpu = 0
    most_free_mem = 0

    for i in range(gpus):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        if info.free > most_free_mem:
            most_free_mem = info.free
            best_gpu = i
    
    return best_gpu
    

def get_top_n_devices(gpus:int = 8, n:int = 2):
    gpu_lst = []
    
    for i in range(gpus):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        gpu_lst.append((i, info.free))
    
    gpu_lst.sort(key = lambda x: x[1])

    return [gpu[0] for gpu in gpu_lst[-n:]]

if __name__ == '__main__':
    for i in range(8):
        get_gpu_mem(i)
        print('\n\n')

    print("Best GPU:", get_best_free_gpu(8))

    print(get_top_n_devices(gpus=8, n=3))


    #print(convert_size(51050315776))