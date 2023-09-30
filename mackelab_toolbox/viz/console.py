"""
Utility for displayed colored/formatted output in the Python console.
"""

class console:
   """
   Usage:

   >>> print(console.BOLD + 'Hello, World!' + console.END)
   
   Source: https://stackoverflow.com/a/17303428
   """
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'