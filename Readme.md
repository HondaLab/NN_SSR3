## NN_SSR2/Li から引き継ぎました．
## NN_SSR2 は削除する予定ですので，こちらで作業してください．

スキッドステアリングロボット(SSR2)をニューラルネットワーク(NN)を用いて自律走行実験する．

teacher data :

https://muroranit-my.sharepoint.com/:u:/g/personal/20043068_mmm_muroran-it_ac_jp/EW3JrT4Qc2FLv4XZDQQAzXkBjjGTF9sr99w809OttwJUJw?e=LjS0YQ


## NN教師データの収集手順（○：走行ロボットの端末の操作　●：データを受信するコンピュータの端末の操作）<br>
1．ロボット側とデータ受信側，両方のフォルダー「modules」の中の「li_sokcket.py」のソースコード内のdata_reciving_terminal（3行目）の値を受信するパソコンのIPアドレスに書き換える．<br>
○2．１のフォルダーで「collention_data.py」を実行する．別のウィンドウでロボットの視点映像が表示される．端末にクリックして，入力できるようにする．画面は暗くなっているが，これはロボットが走行し始めると表示される．以上でロボットの準備完了．<br>
●3．「recv_data.py」を実行する，端末上に表示される質問に答える．以上でデータ受信側のコンピュータも準備完了．<br>
○4．ロボットをラジコン操作する．※操作方法は下記に示す．ロボットの端末に入力行うと，キーボードのボタンを押す瞬間の一次元画像データとモーターの出力をソケット通信でデータ受信側に送信する．データ受信側の端末に受けたデータの数を表示し，設定した数に至ると自動的に停止する．収集したデータは「part_data_in.csv」「part_data_in_include_distance_data.csv」「part_motor_out.csv」として保存される，ファイル名の後ろに番号(手順４で入力した数字)がついてる．<br>
### ロボットの操作方法（a:前進　power +20，z:後退　back power -20，j:左曲がり，l:右曲がり，k:曲がる値を０にする）<br>

## ２回目以降のデータ収集の際の注意点<br>
「rec_data.py」をもう一度実行するとデータは収集される．ファイル名は手順で入力した数字の次のものになる，<br>※データを結合する際，数字が連続されたものでないと結合できないので注意<br>

## 収集したデータの結合方法<br>
１．データ受信側の端末で「intergation_data.py」を実行する．結合したいファイルのstart number と stop number を入力する，結合されたデータは「chainer_data_in.csv」「chainer_data_in_include_distance_data.csv」「chainer_motor_out.csv」に保存される．これは学習用のデータである．このデータは上書きされていくので各自別のフォルダーに移動する．<br>
※NNによる学習するときに学習用のデータが「NN_learning_h1.py」と同じ場所にないと学習できないので注意
２．ロボット側のcollention_data.pyをstopしたら，ロボット視点の動画もフォルダー「temp」にmp4ファイルとして保存される，こちらも上書きされるので学習用のデータと一緒に保存する．<br>

## NNによる学習<br>
１．データを受信したコンピュータで「NN_learning_h1.py」を実行すると，学習データを読み込んで，学習する，学習結果は同じフォルダーに保存される．<br>
２．「send.sh」の４行目のpi@172.16.7.○○○を使用するロボットの番号に書き換える．その後，「send.sh」を実行し学習結果を送信する．<br>
３．学習結果の送信が終わり，ロボット側で「nn_ssr2_h1.py」を実行すると手順２で送信された結果を読み込んで自立走行する．<br>
