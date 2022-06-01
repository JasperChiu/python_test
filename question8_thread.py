import threading
import time


# 子執行緒的工作函數
def thread_job():
    for i in range(5):
        print(f"Child thread:{i}")
        time.sleep(1)


def multi_thread_job(num, num_square):
    print(f"Thread {num} num_square = {num_square**2}")
    time.sleep(1)


if __name__ == "__main__":
    # 建立子執行緒，並指定其要執行的任務(target) - 初始狀態、可執行狀態
    thread = threading.Thread(target=thread_job)
    # start()開始執行該子執行緒 - 執行狀態
    thread.start()
    # 以下為主執行緒，子執行緒會與主執行緒同時處理
    for i in range(3):
        print(f"Main thread:{i}")
        time.sleep(1)

    # join() 等待子執行緒執行完成後才會繼續向下執行，若取消join，會讓All Done先執行出來(子執行緒繼續處理直到完成)
    # 有些情況需要等待所有資料都輸出完，才能繼續向下執行
    thread.join()  # 直到執行完完畢 - 終止狀態
    print("All Done")

    # 建立多執行緒
    threads = []
    for i in range(5):
        # target指向要處理的任務，args接受要傳入的參數
        threads.append(threading.Thread(target=multi_thread_job, args=(i, i)))
        threads[i].start()  # 執行該子執行緒
    for i in range(5):
        threads[i].join()  # 待所有子執行緒執行完畢
    print("All Done")
