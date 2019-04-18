
def delay_return(step: int, num: int, buffer_size:int):
    buf = []
    while True:
        if step % buffer_size == 0:
            result, buf = buf[:], []
            yield result
        else:
            buf.append(num)
            yield []


for i in range(1,13):
    print(delay_return(step=i, num=i, buffer_size=3))
