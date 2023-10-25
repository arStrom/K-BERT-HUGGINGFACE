import time
from datetime import timedelta

def get_time_dif(start_time):
    """
    获取时间间隔
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# print(lines)
start_time = time.time()
time_dif = get_time_dif(start_time)
print("Epoch id: {0:0>2}, Training steps: {1:0>4},  Train Loss: {2:0<6.6f},  Train Acc: {3:0>6.2%},  Train Avg Loss: {4:0<6.6f},  Time: {5}"
        .format(1, 100, 0.001, 0.0750, 0.001, time_dif))