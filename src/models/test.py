# import sys

# sys.path.append('/usr/people/hakvoort/thesiscode/')


from src.models.get_data import get_folds
from src.models.emos import EMOS


X = EMOS(3)

a,b,c,d = get_folds()

print(a[0].date)