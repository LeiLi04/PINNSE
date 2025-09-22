import random 

money = 1000
while money > 0:
    print(f"你现在有 {money} 元。")
    # 玩家下注
    while True:
        try:
            money_bet = int(input('请下注'))
            if money_bet > 0:
                break
            else:
                print("下注金额必须大于0，请重新输入。")
        except ValueError:
            print("请输入一个有效的数字。")
    # 用2个[1,6]的骰子模拟两个随机数的点数
    points_first = random.randrange(1,7) + random.randrange(1,7)
    # 玩家第一次如果摇出来7点或11点，玩家胜， 
    if points_first in (7,11):
        print(f'你摇出来了{points_first}点， 你赢了！')
        money += money_bet
    # 如果第1次摇出来2点，3点，12点，庄家胜
    elif points_first in (2,3,12):
        print(f'你摇出来了{points_first}点， 庄家赢了！')
        money -= money_bet
    # 玩家如果摇出来其他点数游戏继续
    # ------继续重新摇色子------
    elif points_first not in (7,11,2,3,12):
        print(f'你摇出来了{points_first}点， 游戏继续， 请继续摇色子。')
        while True:
            input('按回车继续摇色子')
            points_second = random.randrange(1,7) + random.randrange(1,7)
            print(f'你摇出来了{points_second}点。')
            # 如果玩家摇出来7点， 庄家胜
            if points_second == 7:
                print(f'你摇出来了7点， 庄家赢了！')
                money -= money_bet
                break
            # 如果玩家摇出来第一次摇的点数，玩家胜
            elif points_second == points_first:
                print(f'你摇出来了{points_second}点， 你赢了！')
                money += money_bet
                break
                # 其他点数继续摇色子
            else:
                print(f'你摇出来了{points_second}点， 游戏继续， 请继续摇色子。')

    
    
    
    
    
