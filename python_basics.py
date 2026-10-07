"""Retained repaired small Python functions with guarded list boundaries."""

def round_decimal(number):
    return round(number, 2)

def total_number_of_candies(Alice, Bob, Carol):
    total_candies = Alice + Bob + Carol
    return total_candies % 3

def sign(number):
    if number > 0:
        return 1
    elif number < 0:
        return -1
    else:
        return 0

def hotdog_toppings(ketchup, mustard, onion):
    no_onion = not onion
    all_toppings = ketchup and mustard and onion
    plain_hotdog = not (ketchup or mustard or onion)
    ketchup_or_mustard_only = ketchup != mustard and (not onion)
    return (no_onion, all_toppings, plain_hotdog, ketchup_or_mustard_only)

def second_element_of_list(list1):
    if len(list1) >= 2:
        return list1[1]
    else:
        return None

def second_player(teams):
    return teams[-1][1] if teams and len(teams[-1]) >= 2 else None

def fashionably_late(arrivals, name):
    total_guests = len(arrivals)
    guest_index = arrivals.index(name)
    return guest_index >= total_guests / 2 and guest_index != total_guests - 1

def meal_served_two_days_in_a_row(meals):
    for i in range(1, len(meals)):
        if meals[i] == meals[i - 1]:
            return True
    return False
