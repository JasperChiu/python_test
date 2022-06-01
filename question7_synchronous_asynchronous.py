import asyncio
import time
# 同步(Synchronous): 循序處理，接到一個任務後，需要等到它完成，才能繼續執行下一個任務。
# 非同步(Asynchronous): 平行處理，無需等待第一個任務完成，即可執行其它的任務，只要第一個任務完成了，就回來處理。
# 循序處理測試函數
def sync_print_word():
    print("Hello")
    time.sleep(1)
    print("World")

# 平行處理測試函數
async def print_word():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

async def main():
    await asyncio.gather(print_word(), print_word(), print_word())

start = time.time()  # 紀錄起始時間
asyncio.run(main())  # 執行協同程序
end = time.time()  # 紀錄結束時間
processing_time = end - start  # 計算程式執行時間
print(f"非同步處理執行時間:{processing_time} 秒")

start = time.time()
sync_print_word()  # 相同函式重複執行三次
sync_print_word()
sync_print_word()
end = time.time()
processing_time = end - start
print(f"同步執行時間:{processing_time} 秒")
