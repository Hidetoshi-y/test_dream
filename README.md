# 概要  
DeepDreamを実行する  

# ファイル構成  
input/ DeepDreamで変換したい画像を置く  
.gitignore 関係のないファイルやディレクトリなど
NOTE.md  メモ書き  
README.md ここ  
dream.py 実行されるプログラム  
setup.sh 利用するライブラリを記述  
requirements.txt 利用されたライブラリを記述  
  
# 使い方  
setup.shを実行して環境を作る  
`./setpu.sh`  
  
inputディレクトリにimage.jpgというファイルを置く  
プログラムを実行する  
`python dream.py`  