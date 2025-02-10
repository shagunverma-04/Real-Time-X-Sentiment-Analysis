import json
from pathlib import Path


#create initial cache file 
def create_cache_file():
   initial_cache= {}
   cache_path=Path('tweet_cache.json')

   #create file with empty cache
   with open(cache_path, 'w') as f:
        json.dump(initial_cache, f)

   print(f"Cache file created at: {cache_path.absolute()}")


if __name__ == "__main__":
    create_cache_file()