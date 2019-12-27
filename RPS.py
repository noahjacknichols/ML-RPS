# Perhaps a useless machine
import joblib
import random
import sys
import pandas as pd
import numpy
def getRock():
    ''' need to return a prediction with weights skewed to rock '''
    weights = [43.0,0.0,-2.0,6.0,11.0,26.0,51.0,27.0,-9.0,-2.0,-1.0,-2.0,17.0,20.0,-79.0,-46.0,10.0,6.0,5.0,10.0,-9.0,-10.0,29.0,73.0,-16.0,-7.0,-5.0,-1.0,-13.0,-24.0,-115.0,-61.0,-28.0,1.0,4.0,7.0,18.0,4.0,127.0,46.0,-7.0,-11.0,-7.0,-7.0,-23.0,-14.0,-72.0,-76.0,49.0,9.0,7.0,10.0,17.0,45.0,127.0,41.0,15.0,4.0,10.0,25.0,9.0,13.0,73.0,47.0,0]
    query = []
    for weight in weights:
       query.append(weight * (0.95 + (random.random()*0.05))) 

    return numpy.array(query)
    
def getPaper():
    ''' need to return a prediction with weights skewed to paper '''
    weights = [10.0,4.0,6.0,7.0,-21.0,-59.0,2.0,-6.0,-3.0,-8.0,-5.0,-5.0,20.0,17.0,-1.0,4.0,-13.0,-4.0,-3.0,-1.0,0.0,15.0,-2.0,-3.0,-16.0,-2.0,-4.0,-3.0,15.0,22.0,2.0,3.0,-5.0,-3.0,0.0,0.0,4.0,-6.0,-1.0,-9.0,-7.0,4.0,1.0,-4.0,-5.0,-12.0,1.0,-11.0,7.0,1.0,1.0,2.0,-35.0,-45.0,-6.0,-1.0,14.0,-3.0,-3.0,-3.0,-3.0,-18.0,1.0,4.0,1]
    query = []
    for weight in weights:
        query.append(weight * (0.95 + (random.random()*0.05))) 

    return numpy.array(query) 

def getScissors():
    ''' need to return a prediction with weights skewed to scissors '''
    weights = [0.0,9.0,-6.0,-10.0,-24.0,-10.0,-6.0,8.0,-10.0,-1.0,6.0,-6.0,7.0,-68.0,-18.0,-18.0,14.0,18.0,1.0,1.0,18.0,58.0,44.0,35.0,-3.0,-10.0,4.0,5.0,16.0,17.0,-1.0,-20.0,-4.0,-19.0,-4.0,-2.0,-18.0,-5.0,-9.0,20.0,3.0,27.0,-3.0,2.0,1.0,-49.0,-1.0,-2.0,-15.0,-5.0,-2.0,-5.0,5.0,-5.0,-35.0,-12.0,9.0,1.0,3.0,5.0,6.0,16.0,5.0,12.0,2]
    query = []
    for weight in weights:
        query.append(weight * (0.95 + (random.random()*0.05)))
    
    return numpy.array(query)

def chooseHand():
    roll = random.random() * 100
    if(roll <= 33):
        return getRock()
    elif(roll > 33 and roll <= 66):
        return getPaper()
    else:
        return getScissors()

def pick2Players(players, pastPlayers):
    notPlayedPlayers = list(set(players) - set(pastPlayers))
    if(len(notPlayedPlayers) < 2):
        #one left victorious
        return 'V'
    if(len(notPlayedPlayers) >= 2):
        #more games to play
        return notPlayedPlayers[:2]

#   0 = Rock
#   1 = Paper
#   2 = Scissors
def determineWinner(player1, player2):
    if(player1 == player2):
        #tie
        return -1    
    if(player2 == player1+1):
        return player2
    else:
        return player1
def rollHands(model, players):
    rolled = {}
    dataFile = open('0.csv', 'r')
    # data = dataFile.readline()
    
    for player in players:
        data = chooseHand()
        print(data[:len(data)-1])
        pred = pd.Series(model.predict(data[:len(data)-1]))
        print(pred)
        rolled[player] = int(pred) 

    
    return rolled

def maxScore(score):
    maxScore = 0
    for key in score.keys():
        if(score[key] > maxScore):
            maxScore = score[key]

    return maxScore

def maxScorePlayer(score):
    maxScore = 0
    playerKey = ''
    for key in score.keys():
        if(score[key] > maxScore):
            maxScore = score[key]
            playerKey = key
    return playerKey
'''
players (array): list of names of players to count in rock-paper-scissors.
rounds (int): number of exchanges needed to win the game.
the game goes as follows:
each player rolls a hand (Rock, Paper, Scissors). the winner of each round goes on to face the others of the next round.
This allows a single player to rack up multiple points each round.
'''
def playGame(players, rounds):
    score = {}
    model = joblib.load('MLP_model.joblib')
    for player in players:
        score[player] = 0
    while(maxScore(score) < rounds):
        #play the game
        rolledHands = rollHands(model, players)
        
        pastPlayers = []
        playerRound = pick2Players(players, pastPlayers)
        while(playerRound != 'V'):
            winner = determineWinner(playerRound[0],playerRound[1])
            if(winner == -1):
                #remove them both
                pastPlayers.append(playerRound[0])
                pastPlayers.append(playerRound[1])
            else:
                if(winner == playerRound[0]):
                    #player1 wins
                    pastPlayers.append(playerRound[1])
                    score[playerRound[0]] += 1
                else:
                    #player2 wins
                    pastPlayers.append(playerRound[0])
                    score[playerRound[1]] += 1

    gameWinner = maxScorePlayer(score) 
    print("Congratulations {}, you won!".format(gameWinner))
    

        
    return

def parseInput():
    players = []
    for i in range(1, len(sys.argv) - 2):
        players.append(sys.argv[i])

    rounds = sys.argv[len(sys.argv)-1]
    if(rounds.isalpha()):
        #rounds isn't numeric, so we assume it's a players name.
        players.append(rounds)
        rounds = len(players) * 2 #generate default round value
    return players, int(rounds)
    



def main():
    if(len(sys.argv) >= 3):
        #playable game
        players, rounds = parseInput()
        playGame(players, rounds)
    else:
        print("sorry, your input seems to be wrong. try: RPS.py <player1> <player2> .... <playerX> <rounds>(optional) ")

if __name__ == "__main__":
    main()