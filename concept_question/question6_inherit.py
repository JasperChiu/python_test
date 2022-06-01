# 父類別(基底類別)
class FamilyData:
    def __init__(self):
        self.last_name = "chiu"

    def work_address(self):
        print("桃園市")


# 子類別
class Father(FamilyData):
    def commute_method(self):
        print("開車通勤")

    def call_base_func(self):
        super().work_address()  # super() 為子類別內呼叫父類別下功能的方法


class Son(FamilyData):
    def commute_method(self):
        print("搭火車通勤")

    # 子類別的方法使用相同名稱可以覆蓋取代掉父類別的方法
    def work_address(self):
        print("新北市")


if __name__ == "__main__":
    father_data = Father()
    # Q6.1 呼叫父類別的屬性和方法，子類別內呼叫父類別下功能的方法
    print(father_data.last_name)  # 直接呼叫父類別的屬性
    father_data.work_address()  # 直接呼叫父類別的方法
    father_data.call_base_func()  # 子類別內呼叫父類別下功能的方法

    # Q6.2 子類別中定義了和父類別同名的方法，之後子類別呼叫這個同名方法時，其中的實作內容將會覆蓋掉父類別的同名方法
    son_data = Son()
    print(son_data.last_name)
    son_data.work_address()  # 子類別的方法取代掉父類別的方法
