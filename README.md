# FinetuneLLM-Assignement1C - Aiden Garcia-Rubio
## Instructions
### Environment Setup

In order to run everything, make sure that you have the environment.yml file available in your current directory. Then create the environment using this code:

`conda env create -f environment.yml`

Once it is created you need to activate it with:

`conda activate arc1c`

**NOTE** the environment will need to be activated any time you return.

### Fine Tuning (Task 1)
After that, you can use:

`python LlamaTrain.py`

or

`python MistralTrain.py`

or

`python Phi2Train.py`

To Train the models on the [the python code dataset](https://huggingface.co/datasets/flytech/python-codes-25k). If you want to adjust the models you can also open the model files.

**NOTE** The models can take time to train. Also changing the model type for some of them can cause problems and lead to a "NotImplementedError: Cannot copy out of meta tensor; no data!". They can also cause problems later when you try to use the evaluation code.

### Metric Evaluations
You can open `Evaluation.py` and comment/uncomment the models, the metrics to capture, and the hyperparameters and sizes to execute with. You can find these at the beginning of the code. There is also a variable to set the number of rows to execute. Once the code finishes running the Metric results will be saved to the 'Evaluation_Results.xlsx' file. You may need to wait a minute or two and refresh the explore to see the change.

**Tip** ctrl/ is the shortcut to comment/uncomment lines

**NOTE** Be sure that the fine-tuned model is in the current directory to run

**ALSO NOTE** Evaluation.py can be slow. I recommend setting the iterations to one or a very low amount and commenting on certain parameters if you are testing it out. 

Run `Evaluation.py` with

`python Evaluation.py`

**LAST NOTE** The `Atempts.py` can be used for evaluating the metrics, however it contains a function that allows you to do human evaluation for the models and the hyperparameter tuning. This can take a while even with the iterations set to 1.

**Please see metrics tables and task discussions below (due to time constraints only 1 iteration was used): **

---

## TASK 2

<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8">Model Name</th>
    <th class="tg-9wq8">BLEU Score</th>
    <th class="tg-9wq8">Rouge-L</th>
    <th class="tg-9wq8">BERTScore</th>
    <th class="tg-9wq8">CodeBLEU</th>
    <th class="tg-9wq8">Human Evaluation (20 Samples)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8">LLaMA</td>
    <td class="tg-9wq8">0.07</td>
    <td class="tg-9wq8">'r':0.7692307692307693<br>'p':0.13513513513513514<br>'f':0.22988505492931696</td>
    <td class="tg-9wq8">r:0.8704075813293457<br>p:0.7909638285636902<br>f:0.8287861943244934</td>
    <td class="tg-9wq8">0.50</td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8">Phi-2</td>
    <td class="tg-9wq8">0.16</td>
    <td class="tg-9wq8">'r':0.7692307692307693<br>'p':0.15625<br>'f':0.25974025693371566</td>
    <td class="tg-9wq8">r:0.9000338315963745<br>p:0.8204126358032227<br>f:0.8583808541297913</td>
    <td class="tg-9wq8">0.49</td>
    <td class="tg-9wq8"></td>
  </tr>
  <tr>
    <td class="tg-9wq8">Mistral</td>
    <td class="tg-9wq8">0.08</td>
    <td class="tg-9wq8">'r':0.7692307692307693<br>'p':0.3333333333333333<br>'f':0.46511627485127094</td>
    <td class="tg-9wq8">r:0.9005714654922485<br>p:0.7690525054931641<br>f:0.8296319842338562</td>
    <td class="tg-9wq8">0.51</td>
    <td class="tg-9wq8"></td>
  </tr>
</tbody>
</table>

**Write a discussion (4-5 Lines) explaining the comparison between two models. Moreover, compare the metrics and discuss which metrics are more appropriate compared to human evaluation. **

The Mistral and Phi-2 models tend to outperform the Llama model in most metrics. Phi-2 seems to perform the best out them. While Llama did answer the questions the length of the answer could be long. This would make it lose points because it often added answers to instructions it was not given. It would not stop when it completed the question. This was most likely due to my limited training with the integrated runs. This problem did occur in the Phi-2 and Mistral tests but not as frequently as Llamas.


## Task 3

<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix">Model Name</th>
    <th class="tg-nrix">Hyperparameter</th>
    <th class="tg-nrix">Size</th>
    <th class="tg-nrix">BLEU</th>
    <th class="tg-nrix">Rouge-L</th>
    <th class="tg-nrix">BERTScore</th>
    <th class="tg-nrix">CodeBLEU</th>
    <th class="tg-nrix">Human Evaluation (20 Samples)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="12">LLaMA</td>
    <td class="tg-nrix" rowspan="4">Top_k</td>
    <td class="tg-nrix">2</td>
    <td class="tg-nrix">0.06</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.09615384615384616<br>'f':0.1709401689648623</td>
    <td class="tg-nrix">r:0.8854020833969116<br>p:0.7797147035598755<br>f:0.8292043209075928</td>
    <td class="tg-nrix">0.51</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">4</td>
    <td class="tg-nrix">0.07</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.10526315789473684<br>'f':0.1851851830675583</td>
    <td class="tg-nrix">r:0.86870276927948<br>p:0.7961644530296326<br>f:0.8308533430099487<br></td>
    <td class="tg-nrix">0.51</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">6</td>
    <td class="tg-nrix">0.06</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.0970873786407767<br>'f':0.17241379111325805</td>
    <td class="tg-nrix">r:0.8789341449737549<br>p:0.7905920743942261<br>f:0.8324257731437683</td>
    <td class="tg-nrix">0.53</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">8</td>
    <td class="tg-nrix">0.09</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.09174311926605505<br>'f':0.16393442432545016</td>
    <td class="tg-nrix">r:0.8977103233337402<br>p:0.7937819957733154<br>f:0.8425533771514893</td>
    <td class="tg-nrix">0.53</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="4">Beam_size</td>
    <td class="tg-nrix">2</td>
    <td class="tg-nrix">0.12</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.21739130434782608<br>'f':0.33898304741166335</td>
    <td class="tg-nrix">r:0.8756498098373413<br>p:0.8177005052566528<br>f:0.8456835746765137</td>
    <td class="tg-nrix">0.51</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">3</td>
    <td class="tg-nrix">0.06</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.19607843137254902<br>'f':0.31249999676269535</td>
    <td class="tg-nrix">r:0.8704752326011658<br>p:0.7936128377914429<br>f:0.8302689790725708</td>
    <td class="tg-nrix">0.49</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">4</td>
    <td class="tg-nrix">0.11</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.125<br>'f':0.21505376103595794</td>
    <td class="tg-nrix">r:0.8957557678222656<br>p:0.8134111762046814<br>f:0.8525997996330261</td>
    <td class="tg-nrix">0.52</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">5</td>
    <td class="tg-nrix">0.06</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.19607843137254902<br>'f':0.31249999676269535</td>
    <td class="tg-nrix">r:0.8704752326011658<br>p:0.7936128377914429<br>f:0.8302689790725708</td>
    <td class="tg-nrix">0.49</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="4">Temperature</td>
    <td class="tg-nrix">0.10</td>
    <td class="tg-nrix">0.07</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.09900990099009901<br>'f':0.17543859447060636</td>
    <td class="tg-nrix">r:0.8701425790786743<br>p:0.783594012260437<br>f:0.8246034979820251</td>
    <td class="tg-nrix">0.51</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">0.25</td>
    <td class="tg-nrix">0.08</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.2127659574468085<br>'f':0.3333333299388889</td>
    <td class="tg-nrix">r:0.872684121131897<br>p:0.7834901809692383<br>f:0.8256853222846985</td>
    <td class="tg-nrix">0.50</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">0.50</td>
    <td class="tg-nrix">0.07</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.10309278350515463<br>'f':0.1818181797338843</td>
    <td class="tg-nrix">r:0.8791018128395081<br>p:0.7931342720985413<br>f:0.8339083194732666</td>
    <td class="tg-nrix">0.53</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">0.75</td>
    <td class="tg-nrix">0.03</td>
    <td class="tg-nrix">'r':0.38461538461538464<br>'p':0.06172839506172839<br>'f':0.1063829763399729</td>
    <td class="tg-nrix">r:0.8710653781890869<br>p:0.7935007810592651<br>f:0.830475926399231</td>
    <td class="tg-nrix">0.35</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="12">Phi-2</td>
    <td class="tg-nrix" rowspan="4">Top_k</td>
    <td class="tg-nrix">2</td>
    <td class="tg-nrix">0.04</td>
    <td class="tg-nrix">'r':0.5384615384615384<br>'p':0.05511811023622047<br>'f':0.09999999831530614</td>
    <td class="tg-nrix">r:0.8756164312362671<br>p:0.7857187986373901<br>f:0.8282353281974792</td>
    <td class="tg-nrix">0.40</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">4</td>
    <td class="tg-nrix">0.09</td>
    <td class="tg-nrix">'r':0.46153846153846156<br>'p':0.1111111111111111<br>'f':0.17910447448429498</td>
    <td class="tg-nrix">r:0.8901036381721497<br>p:0.8321672677993774<br>f:0.8601608872413635</td>
    <td class="tg-nrix">0.40</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">6</td>
    <td class="tg-nrix">0.49</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.3333333333333333<br>'f':0.46511627485127094</td>
    <td class="tg-nrix">r:0.9121323227882385<br>p:0.8623875379562378<br>f:0.8865626454353333</td>
    <td class="tg-nrix">0.56</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">8</td>
    <td class="tg-nrix">0.04</td>
    <td class="tg-nrix">'r':0.38461538461538464<br>'p':0.078125<br>'f':0.12987012706358583</td>
    <td class="tg-nrix">r:0.8680506944656372<br>p:0.8166894912719727<br>f:0.8415871858596802</td>
    <td class="tg-nrix">0.31</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="4">Beam_size</td>
    <td class="tg-nrix">2</td>
    <td class="tg-nrix">0.14</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.13157894736842105<br>'f':0.224719098628961</td>
    <td class="tg-nrix">r:0.8951790928840637<br>p:0.8144630193710327<br>f:0.8529156446456909</td>
    <td class="tg-nrix">0.48</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">3</td>
    <td class="tg-nrix">0.15</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.13333333333333333<br>'f':0.2272727247546488</td>
    <td class="tg-nrix">r:0.8948008418083191<br>p:0.81748366355896<br>f:0.854396641254425</td>
    <td class="tg-nrix">0.48</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">4</td>
    <td class="tg-nrix">0.13</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.12195121951219512<br>'f':0.21052631342714684</td>
    <td class="tg-nrix">r:0.8956357836723328<br>p:0.8088359832763672<br>f:0.8500257134437561</td>
    <td class="tg-nrix">0.48</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">5</td>
    <td class="tg-nrix">0.13</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.12195121951219512<br>'f':0.21052631342714684</td>
    <td class="tg-nrix">r:0.8956357836723328<br>p:0.8088359832763672<br>f:0.8500257134437561</td>
    <td class="tg-nrix">0.48</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="4">Temperature</td>
    <td class="tg-nrix">0.10</td>
    <td class="tg-nrix">0.16</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.15384615384615385<br>'f':0.25641025363247866</td>
    <td class="tg-nrix">r:0.899850606918335<br>p:0.8193579316139221<br>f:0.8577200174331665</td>
    <td class="tg-nrix">0.49</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">0.25</td>
    <td class="tg-nrix">0.15</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.14492753623188406<br>'f':0.24390243635633554</td>
    <td class="tg-nrix">r:0.8996610641479492<br>p:0.8151978254318237<br>f:0.8553493618965149</td>
    <td class="tg-nrix">0.51</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">0.50</td>
    <td class="tg-nrix">0.15</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.14084507042253522<br>'f':0.23809523547902497</td>
    <td class="tg-nrix">r:0.8975547552108765<br>p:0.8151139616966248<br>f:0.854350209236145</td>
    <td class="tg-nrix">0.48</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">0.75</td>
    <td class="tg-nrix">0.09</td>
    <td class="tg-nrix">'r':0.5384615384615384<br>'p':0.10144927536231885<br>'f':0.1707317046490185</td>
    <td class="tg-nrix">r:0.8936412334442139<br>p:0.8175271153450012<br>f:0.8538913726806641</td>
    <td class="tg-nrix">0.38</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="12">Mistral</td>
    <td class="tg-nrix" rowspan="4">Top_k</td>
    <td class="tg-nrix">2</td>
    <td class="tg-nrix">0.07</td>
    <td class="tg-nrix">'r':0.6923076923076923<br>'p':0.20454545454545456<br>'f':0.31578947016312714</td>
    <td class="tg-nrix">r:0.8968472480773926<br>p:0.7445937991142273<br>f:0.8136593103408813</td>
    <td class="tg-nrix">0.47</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">4</td>
    <td class="tg-nrix">0.08</td>
    <td class="tg-nrix">'r':0.46153846153846156<br>'p':0.08695652173913043<br>'f':0.14634146074657944</td>
    <td class="tg-nrix">r:0.8859530687332153<br>p:0.8228813409805298<br>f:0.8532532453536987</td>
    <td class="tg-nrix">0.39</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">6</td>
    <td class="tg-nrix">0.09</td>
    <td class="tg-nrix">'r':0.6923076923076923<br>'p':0.09278350515463918<br>'f':0.16363636155206615</td>
    <td class="tg-nrix">r:0.8980460166931152<br>p:0.7940094470977783<br>f:0.8428294062614441</td>
    <td class="tg-nrix">0.46</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">8</td>
    <td class="tg-nrix">0.11</td>
    <td class="tg-nrix">'r':0.5384615384615384<br>'p':0.11666666666666667<br>'f':0.19178081899042976</td>
    <td class="tg-nrix">r:0.8954402208328247<br>p:0.826213002204895<br>f:0.8594347834587097</td>
    <td class="tg-nrix">0.37</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="4">Beam_size</td>
    <td class="tg-nrix">2</td>
    <td class="tg-nrix">0.07</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.2702702702702703<br>'f':0.39999999615200005</td>
    <td class="tg-nrix">r:0.8997339010238647<br>p:0.7745327353477478<br>f:0.832452118396759</td>
    <td class="tg-nrix">0.51</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">3</td>
    <td class="tg-nrix">0.25</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.2<br>'f':0.31746031418493326</td>
    <td class="tg-nrix">r:0.906930685043335<br>p:0.8348859548568726<br>f:0.8694183230400085</td>
    <td class="tg-nrix">0.45</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">4</td>
    <td class="tg-nrix">0.25</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.2<br>'f':0.31746031418493326</td>
    <td class="tg-nrix">r:0.906930685043335<br>p:0.8348859548568726<br>f:0.8694183230400085</td>
    <td class="tg-nrix">0.45</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">5</td>
    <td class="tg-nrix">0.11</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.1724137931034483<br>'f':0.28169013785360053</td>
    <td class="tg-nrix">r:0.8840132355690002<br>p:0.8102754354476929<br>f:0.8455397486686707</td>
    <td class="tg-nrix">0.52</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="4">Temperature</td>
    <td class="tg-nrix">0.10</td>
    <td class="tg-nrix">0.08</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.3225806451612903<br>'f':0.45454545038223143</td>
    <td class="tg-nrix">r:0.900871992111206<br>p:0.7710911631584167<br>f:0.8309445977210999</td>
    <td class="tg-nrix">0.51</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">0.25</td>
    <td class="tg-nrix">0.07</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.3225806451612903<br>'f':0.45454545038223143</td>
    <td class="tg-nrix">r:0.8634172081947327<br>p:0.818548858165741<br>f:0.8403845429420471</td>
    <td class="tg-nrix">0.48</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">0.50</td>
    <td class="tg-nrix">0.21</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.17543859649122806<br>'f':0.28571428268979593</td>
    <td class="tg-nrix">r:0.9032034873962402<br>p:0.8294703960418701<br>f:0.8647680878639221</td>
    <td class="tg-nrix">0.52</td>
    <td class="tg-nrix"></td>
  </tr>
  <tr>
    <td class="tg-nrix">0.75</td>
    <td class="tg-nrix">0.26</td>
    <td class="tg-nrix">'r':0.7692307692307693<br>'p':0.20408163265306123<br>'f':0.32258064184703433</td>
    <td class="tg-nrix">r:0.9082292914390564<br>p:0.8371737003326416<br>f:0.8712551593780518</td>
    <td class="tg-nrix">0.51</td>
    <td class="tg-nrix"></td>
  </tr>
</tbody>
</table>

**Write another discussion explaining how the hyperparameters affect the different metrics of LLaMA and Phi-2 (4-5 Lines). **
Increasing the top_k parameter Would cause the BLEU Score and CodeBLEU to increase. This is most likely because it allows the model to select from a bigger sampling of tokens. Increasing the beam size hyperparameter seems to raise the scores across the board for every metric except for the recall metric for Rouge-L recall. Finally, Temperature seems cause the metrics to increase and decrease depending on its size. When the size was changed from .25 and .5 some of the metrics showed improvement implying that there is an optimal configuration for it. If it is too high then it opens the door for low probability tokens and if it is too low it can cause it to rely too much on more probable tokens based on its training data, resulting in less variation in generated text.
