from CommonUtil import Util

def function_test(x , y):
    return x*y , x+y

if __name__ == '__main__':
    # function
    a , b = function_test(3,5)
    print(a)
    print(b)

    # # call class
    util = Util()
    data = util.GetRoundDownFloat(0.55558888,4)
    print(data)
