## NN_SSR2/Li から引き継ぎました．NN_SSR2 は削除する予定ですので，こちらで作業してください．

スキッドステアリングロボット(SSR3)をニューラルネットワーク(NN)を用いて
自律走行実験します．

前提として，ロボットとPCは[wifiでネットワーク接続](https://github.com/HondaLab/Robot-Intelligence/wiki/wifi%E3%81%AE%E8%A8%AD%E5%AE%9A)されている必要があります．

手順は大きく別れた３段階あります．
1. 教師データの収集
2. NNの学習
3. 自律走行


## 1.教師データの収集
ロボットとPCの双方でプログラム(Python)を実行します．
（○：走行ロボットの端末の操作　●：データを受信するコンピュータの端末の操作）<br>
### a) IPアドレス (○ ● )
ロボット側とデータ受信側，両方のフォルダー'modules'の中の'socket.py'の
ソースコード内のrecv_addr（3行目）の値を受信するパソコンのIPアドレスに
書き換えます．<br>

### b) ロボットでプログラムを実行する(○ ) 
'collention_data.py'を実行します．
別のウィンドウでロボットの視点映像が表示されます．
端末をクリックして，キー入力できるようにしてください．
ロボットが走行し始めると表示されます．

### c) PCで受信プログラムを実行する(● ) 
'recv_data.py'を実行します．
端末上に表示される質問に答えてください．

### d) ロボットをラジコン操作する(○ ) 
※ 操作方法は下記に示します．

ロボットの端末に入力行うと，キーボードのボタンを押す瞬間の一次元画像データと
モーターの出力をソケット通信でデータ受信側に送信します．
データ受信側の端末に受けたデータの数が表示され，
設定した数に至ると自動的に停止します．
収集したデータは

 - part_data_in.csv
 - part_data_in_include_distance_data.csv
 - part_motor_out.csv

として保存されます．
ファイル名の後ろに番号(手順3で入力した数字)がつきます．<br>

### ロボットの操作方法
（a:前進　power +20，z:後退　back power -20，j:左曲がり，l:右曲がり，k:曲がる値を０にする）<br>

### ２回目以降のデータ収集の際の注意点<br>
'recv_data.py'をもう一度実行するとデータは収集されます．
ファイル名は手順c)で入力した数字の次のものになります．<br>
※データを結合する際，数字が連続されたものでないと結合できないので注意<br>

### 収集したデータの結合方法<br>
 1.データ受信側の端末で'intergation_data.py'を実行します．
結合したいファイルのstart number と stop number を入力します．
結合されたデータは

 - chainer_data_in.csv 
 - chainer_data_in_include_distance_data.csv
 - chainer_motor_out.csv

に保存されます．
これが学習用のデータです．
このデータは上書きされていくので保存したい場合は,
各自別のフォルダーに移動してください．<br>
※NNによる学習するときに学習用のデータが'NN_learning_h1.py'と同じ場所にないと学習できないので注意

 2.ロボット側の'collention_data.py'をstop します．

ロボット視点の動画もフォルダー/tmpにmp4ファイルとして保存されます．
こちらも上書きされるので学習用のデータと一緒に移動して保存してください．<br>

## 2.NNの学習<br>
1.データを受信したコンピュータで 'NN_learning_h1.py'を実行すると，
学習データを読み込んで，学習が実行されます．
学習結果は同じフォルダーに保存されます．<br>
2.'send.sh'の４行目のpi@172.16.7.○○○を使用するロボットのアドレスに書き換えます．
その後，'send.sh'を実行し学習結果をロボットに送信します．<br>


## 3.自律走行．
学習結果の送信が終わり，ロボット側で'nn_ssr2_h1.py'を実行すると
上の手順2.で送信された結果を読み込んで自律走行が開始します．<br>
