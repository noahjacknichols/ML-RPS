# Perhaps a useless machine
import joblib
import pickle
import random
import sys
import pandas as pd
import numpy





def pick2Players(players, pastPlayers):
    notPlayedPlayers = list(set(players) - set(pastPlayers))
    if(len(notPlayedPlayers) < 2):
        #one left victorious
        return 'V'
    if(len(notPlayedPlayers) >= 2):
        #more games to play
        return notPlayedPlayers[:2]

def getHand(roll):
    if(roll == 0):
        return 'Rock'
    elif(roll == 1):
        return 'Paper'
    else:
        return 'Scissors'
#   0 = Rock
#   1 = Paper
#   2 = Scissors
def determineWinner(player1, player2):
    print("Players rolled:",getHand(player1),",",getHand(player2))
    if(player1 == player2):
        #tie
        return -1    
    if(player2 == (player1+1)%3):
        return player2
    else:
        return player1
def rollHands(model, randomHands,  players):
    rolled = {}
    
    for player in players:
        row = randomHands.sample(n=1)
        # print(row)
        pred = model.predict(randomHands.sample(n=1)) 
        # print(pred)
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
    names = [str(i) for i in range(65)]
    randomHands = pd.read_csv('random_data.csv', names = names)
    df = pd.DataFrame(randomHands, columns=names[:64])
    score = {}
    with open('pickle_model.pkl', 'rb') as file:
        model = pickle.load(file) 
    for player in players:
        score[player] = 0
    while(maxScore(score) < rounds):
        #play the game
        
        rolledHands = rollHands(model,df, players)
        pastPlayers = []
        playerRound = pick2Players(players, pastPlayers)
        while(playerRound != 'V'):
            print("score is:",score)
            winner = determineWinner(rolledHands[playerRound[0]],rolledHands[playerRound[1]])
            if(winner == -1):
                #remove them both
                pastPlayers.append(playerRound[0])
                pastPlayers.append(playerRound[1])
            else:
                if(winner == rolledHands[playerRound[0]]):
                    #player1 wins
                    pastPlayers.append(playerRound[1])
                    score[playerRound[0]] += 1
                else:
                    #player2 wins
                    pastPlayers.append(playerRound[0])
                    score[playerRound[1]] += 1
            playerRound = pick2Players(players, pastPlayers)
            # rolledHands = rollHands(model,df,players)

    gameWinner = maxScorePlayer(score) 
    print("Congratulations {}, you won!".format(gameWinner))
    print("final scores:",score)
    

        
    return

def parseInput():
    players = []
    for i in range(1, len(sys.argv) - 1):
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