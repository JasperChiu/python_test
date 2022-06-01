class Test:
    def test_func_public(self):  # 公共的類型
        print("Public")

    def _test_func_protected(self):  # 保護的類型
        print("Protected")

    def __test_func_private(self):  # 私有的類型
        print("Private")

    def func_use_private_func(self):
        # 同一類內可以調用類內私有函式、保護函式
        self._test_func_protected()
        self.__test_func_private()


if __name__ == "__main__":
    test = Test()
    test.test_func_public()  # 呼叫類內公共的函式
    test._test_func_protected()  # 呼叫類內保護的函式，(預設不會顯示，但可以強制叫出來)
    try:
        test.__test_func_private()  # 嘗試呼叫類內私有的函式
    except:
        print("調用私有的函式失敗")
    test.func_use_private_func()  # 同一類內可以調用類內私有函式、保護函式
