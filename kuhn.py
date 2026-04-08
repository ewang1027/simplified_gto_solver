# K > Q > J, can use 2, 1, 0, respectively
cards = ['J', 'Q', 'K']
num_actions = 2

# depends on the action before, 0 is check/fold, 1 is bet/call 
Pass = 0
Bet = 1

class KuhnState:
    def __init__(self, cards, history = None):
        self.cards = cards  # list of cards, cards[0] is player 0's card, cards[1] is player 1's card
        self.history = history if history is not None else []  #list of actions

    def current_player(self): 
        """
        can determine whos turn it is by actions taken
        if history = 0, then it must be p0's turn
        if history = 1, then it must be p1's turn
        if history = 2, then it depends

        it goes back to P0 when history = 2 when it goes check -> bet
        """
        if len(self.history) == 0:
            return 0
        elif len(self.history) == 1:
             return 1
        else:
            return 0

    def is_terminal(self):
        #determines if the game is over
        h = self.history
        if len(h) < 2:
            return False
        elif len(h) == 2:
            if h[0] == Pass and h[1] == Bet:
                return False
            else:
                return True
        else:
            return True #bc len == 3 is always terminal


    def payout(self):
        """
        returns payout for p0, only run once terminal == true 
        """

        h = self.history
        p0_card = self.cards[0]
        p1_card = self.cards[1]
        p0_wins = (p0_card > p1_card)

        if h == [Pass, Pass]: #both players check
            return 1 if p0_wins else -1
        if h == [Bet, Pass]: #p0 bets, p1 folds
            return 1
        if h == [Pass, Bet, Pass]: #p0 checks, p1 bets, p0 folds
            return -1
        if h == [Pass, Bet, Bet]: #p0 checks, p1 bets, p0 calls
            return 2 if p0_wins else -2
        if h == [Bet, Bet]: #p0 bets, p1 calls  
            return 2 if p0_wins else -2
        raise ValueError("Invalid history: {h}")
    
    def info_set(self):
        """
        returns the information set for the current player, which is a tuple of (card, history)
        """
        card = cards[self.cards[self.current_player()]]
        if not self.history:
            return card
        action_str = ''.join('b' if a == Bet else 'c' for a in self.history)
        return f"{card}|{action_str}"
    
    def use_action(self, action):
        #returns new state with action added, make sure not to ruin original state
        return KuhnState(
            cards = self.cards,
            history = self.history + [action]
        )
    
    def get_actions(self):
        #only two actions, if not terminal
        return [Pass, Bet]