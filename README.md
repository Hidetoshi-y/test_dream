# 概要  
DeepDreamを実行する  

# ファイル構成  
input/ DeepDreamで変換したい画像を置く  
.gitignore 関係のないファイルやディレクトリなど
NOTE.md  メモ書き  
README.md ここ  
dream.py 実行されるプログラム  
requirements.txt 利用されたライブラリを記述  
setup.sh 利用するライブラリを記述  
sinclude.sh enrootで利用するコンテナイメージを指定するファイル

  
# 使い方  
setup.shを実行して環境を作る  
`./setpu.sh`  
  
inputディレクトリにimage.jpgというファイルを置く  
プログラムを実行する  
`python dream.py`  