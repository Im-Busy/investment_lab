# TwitterDistantSupervision09

> *Source PDF: TwitterDistantSupervision09.pdf*
> *Extraction: Combined — markitdown raw text (base) + previous extraction supplements*

---

| Twitter |                    | Sentiment |     | Classification |     |                    | using   |     | Distant |                    | Supervision |     |     |     |
| ------- | ------------------ | --------- | --- | -------------- | --- | ------------------ | ------- | --- | ------- | ------------------ | ----------- | --- | --- | --- |
|         |                    | Alec      | Go  |                |     | Richa              | Bhayani |     |         | Lei                | Huang       |     |     |     |
|         | StanfordUniversity |           |     |                |     | StanfordUniversity |         |     |         | StanfordUniversity |             |     |     |     |
|         | Stanford,CA94305   |           |     |                |     | Stanford,CA94305   |         |     |         | Stanford,CA94305   |             |     |     |     |
alecmgo@stanford.edu rbhayani@stanford.edu leirocky@stanford.edu
ABSTRACT
|     |     |     |     |     |     |     | Consumers | can | use sentiment |     | analysis | to  | research | products |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | ------------- | --- | -------- | --- | -------- | -------- |
We introduce a novel approach for automatically classify- orservicesbeforemakingapurchase. Marketerscanusethis
ing the sentiment of Twitter messages. These messages are to research public opinion of their company and products,
classified as either positive or negative with respect to a or to analyze customer satisfaction. Organizations can also
query term. This is useful for consumers who want to re- usethistogathercriticalfeedbackaboutproblemsinnewly
search the sentiment of products before purchase, or com- released products.
| panies that | want | to monitor | the | public | sentiment | of their |     |     |     |     |     |     |     |     |
| ----------- | ---- | ---------- | --- | ------ | --------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
Therehasbeenalargeamountofresearchintheareaofsen-
| brands. | There | is no previous |     | research | on classifying | sen- |     |     |     |     |     |     |     |     |
| ------- | ----- | -------------- | --- | -------- | -------------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
timent of messages on microblogging services like Twitter. timentclassification. Traditionallymostofithasfocusedon
We present the results of machine learning algorithms for classifyinglargerpiecesoftext,likereviews[9]. Tweets(and
classifying the sentiment of Twitter messages using distant microblogs in general) are different from reviews primarily
supervision. OurtrainingdataconsistsofTwittermessages because of their purpose: while reviews represent summa-
withemoticons,whichareusedasnoisylabels. Thistypeof rized thoughts of authors, tweets are more casual and lim-
training data is abundantly available and can be obtained ited to 140 characters of text. Generally, tweets are not as
through automated means. We show that machine learn- thoughtfullycomposedasreviews. Yet,theystilloffercom-
ingalgorithms(NaiveBayes,MaximumEntropy,andSVM) panies an additional avenue to gather feedback. There has
haveaccuracyabove80%whentrainedwithemoticondata. beensomeworkbyresearchersintheareaofphraseleveland
|            |         |                |                   |          |                |           | sentencelevelsentimentclassificationrecently[11]. |              |     |            |          |     |      | Previous |
| ---------- | ------- | -------------- | ----------------- | -------- | -------------- | --------- | ------------------------------------------------- | ------------ | --- | ---------- | -------- | --- | ---- | -------- |
| This paper | also    | describes      | the preprocessing |          | steps          | needed in |                                                   |              |     |            |          |     |      |          |
|            |         |                |                   |          |                |           | research                                          | on analyzing |     | blog posts | includes |     | [6]. |          |
| order to   | achieve | high accuracy. |                   | The main | contribution   | of        |                                                   |              |     |            |          |     |      |          |
| this paper | is the  | idea of        | using             | tweets   | with emoticons | for       |                                                   |              |     |            |          |     |      |          |
distant supervised learning. Previous research in sentiment analysis like Pang et al. [9]
haveanalyzedtheperformanceofdifferentclassifiersonmovie
CategoriesandSubjectDescriptors reviews. The work of Pang et al. has served as a baseline
andmanyauthorshaveusedthetechniquesprovidedintheir
| I.2[Artificial | Intelligence]: |     | NaturalLanguageProcessing |     |     |     |              |           |          |          |      |         |             |          |
| -------------- | -------------- | --- | ------------------------- | --- | --- | --- | ------------ | --------- | -------- | -------- | ---- | ------- | ----------- | -------- |
|                |                |     |                           |     |     |     | paper across | different |          | domains. | Pang | et      | al. also    | make use |
|                |                |     |                           |     |     |     | of a similar | idea      | as ours, | using    | star | ratings | as polarity | sig-     |
GeneralTerms
|     |     |     |     |     |     |     | nals in | their training |     | data. | We show | that | we can | produce |
| --- | --- | --- | --- | --- | --- | --- | ------- | -------------- | --- | ----- | ------- | ---- | ------ | ------- |
Algorithms comparable results on tweets with distant supervision.
Keywords
|     |     |     |     |     |     |     | In order | to train | a classifier, |     | supervised | learning | usually | re- |
| --- | --- | --- | --- | --- | --- | --- | -------- | -------- | ------------- | --- | ---------- | -------- | ------- | --- |
Twitter, sentiment analysis, sentiment classification quires hand-labeled training data. With the large range of
|     |     |     |     |     |     |     | topics discussed |     | on Twitter, |     | it would | be  | very difficult | to  |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ----------- | --- | -------- | --- | -------------- | --- |
1. INTRODUCTION manually collect enough data to train a sentiment classifier
|               |                  |                   |           |                |              |              | for tweets.                            | Our      | solution | is           | to use    | distant | supervision,    | in        |
| ------------- | ---------------- | ----------------- | --------- | -------------- | ------------ | ------------ | -------------------------------------- | -------- | -------- | ------------ | --------- | ------- | --------------- | --------- |
| Twitter       | is a popular     | microblogging     |           | service        | where        | users cre-   |                                        |          |          |              |           |         |                 |           |
|               |                  |                   |           |                |              |              | which our                              | training | data     | consists     | of        | tweets  | with emoticons. |           |
| ate status    | messages         | (called“tweets”). |           |                | These tweets | some-        |                                        |          |          |              |           |         |                 |           |
|               |                  |                   |           |                |              |              | ThisapproachwasintroducedbyRead[10].   |          |          |              |           |         | Theemoticons    |           |
| times express | opinions         | about             | different |                | topics.      | We propose   |                                        |          |          |              |           |         |                 |           |
|               |                  |                   |           |                |              |              | serve as                               | noisy    | labels.  | For example, |           | :) in   | a tweet         | indicates |
| a method      | to automatically |                   | extract   | sentiment      |              | (positive or |                                        |          |          |              |           |         |                 |           |
|               |                  |                   |           |                |              |              | that the                               | tweet    | contains | positive     | sentiment |         | and :(          | indicates |
| negative)     | from a           | tweet.            | This      | is very useful | because      | it al-       |                                        |          |          |              |           |         |                 |           |
|               |                  |                   |           |                |              |              | thatthetweetcontainsnegativesentiment. |          |          |              |           |         | Withthehelpof   |           |
lowsfeedbacktobeaggregatedwithoutmanualintervention.
theTwitterAPI,itiseasytoextractlargeamountsoftweets
|     |     |     |     |     |     |     | with emoticons |           | in them. | This        | is a      | significant | improvement   |          |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --------- | -------- | ----------- | --------- | ----------- | ------------- | -------- |
|     |     |     |     |     |     |     | over the       | many      | hours    | it may      | otherwise | take        | to hand-label |          |
|     |     |     |     |     |     |     | training       | data.     | We run   | classifiers | trained   |             | on emoticon   | data     |
|     |     |     |     |     |     |     | against        | a test    | set of   | tweets      | (which    | may         | or may        | not have |
|     |     |     |     |     |     |     | emoticons      | in them). |          |             |           |             |               |          |
Wepresenttheresultsofourexperimentsandourthoughts
|     |     |     |     |     |     |     | onhowtofurtherimproveresults. |     |     |     | Tohelpvisualizetheutil- |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | ----------------------- | --- | --- | --- |

ity of a Twitter-based sentiment analysis tool, we also have frameworkallowsustoeasilytryoutdifferentcombinations
awebapplicationwithourclassifiers1. Thiscanbeusedby of classifiers and feature extractors.
individualsandcompaniesthatmaywanttoresearchsenti-
| ment on | any topic. |     |     |     |     |     | 2.1 QueryTerm                     |     |     |     |                    |     |     |
| ------- | ---------- | --- | --- | --- | --- | --- | --------------------------------- | --- | --- | --- | ------------------ | --- | --- |
|         |            |     |     |     |     |     | Wenormalizetheeffectofqueryterms. |     |     |     | Table1listsexample |     |     |
1.1 DefiningSentiment
|     |     |     |     |     |     |     | query termsalongwithcorrespondingtweets. |     |     |     |     | Our assump- |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --- | ----------- | --- |
For the purposes of our research, we define sentiment to be tionisthatusersprefertoperformsentimentanalysisabout
“apersonalpositiveornegativefeeling.”Table1showssome aproductandnotof aproduct. Whenauserentersaquery
examples. ‘XYZ’, we normalize the sentiment carried by ‘XYZ’ itself.
|     |     |     |     |     |     |     | Forexample,thetweetXYZ |     |     |     | is hardly interesting | shouldbe |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | --- | --------------------- | -------- | --- |
Manytimesitisunclearifatweetcontainsasentiment. For classifiedasnegative. Iftheword“XYZ”byitselfhasapos-
these cases, we use the following litmus test: If the tweet itive sentiment, it would bias the results. Our approach is
couldeverappearasafrontpagenewspaperheadlineorasa to represent each query term as a QUERY TERM equiva-
| sentence | in Wikipedia, |     | then it | belongs in | the neutral | class. |     |     |     |     |     |     |     |
| -------- | ------------- | --- | ------- | ---------- | ----------- | ------ | --- | --- | --- | --- | --- | --- | --- |
lenceclass,whichallowsustonormalizetheeffectithason
| For example, | the | following | tweet | is considered |     | neutral be- | classification. |     |     |     |     |     |     |
| ------------ | --- | --------- | ----- | ------------- | --- | ----------- | --------------- | --- | --- | --- | --- | --- | --- |
causeitcouldhaveappearedasanewspaperheadline,even
thoughitprojectsanoverallnegativefeelingaboutGeneral 2.2 Emoticons
| Motors: | RT @Finance |      | Info Bankruptcy      |     | filing could | put GM |            |            |         |         |                  |         |           |
| ------- | ----------- | ---- | -------------------- | --- | ------------ | ------ | ---------- | ---------- | ------- | ------- | ---------------- | ------- | --------- |
|         |             |      |                      |     |              |        | Since the  | training   | process | makes   | use of emoticons | as      | noisy     |
| on road | to profits  | (AP) | http://cli.gs/9ua6Sb |     | #Finance.    | In     |            |            |         |         |                  |         |           |
|         |             |      |                      |     |              |        | labels, it | is crucial | to      | discuss | the role they    | play in | classifi- |
thisresearch,wedonotconsiderneutraltweetsinourtrain-
cation. Wewilldiscussindetailourtrainingandtestsetin
| ingortestingdata. |              | Weonlyusepositiveornegativetweets. |             |     |                 |        |                |               |          |       |                   |        |        |
| ----------------- | ------------ | ---------------------------------- | ----------- | --- | --------------- | ------ | -------------- | ------------- | -------- | ----- | ----------------- | ------ | ------ |
|                   |              |                                    |             |     |                 |        | the Evaluation |               | section. |       |                   |        |        |
| Many tweets       | do           | not have                           | sentiment,  | so  | it is a current | limi-  |                |               |          |       |                   |        |        |
| tation of         | our research | to                                 | not include | the | neutral         | class. |                |               |          |       |                   |        |        |
|                   |              |                                    |             |     |                 |        | We strip       | the emoticons |          | out   | from our training | data.  | If we  |
|                   |              |                                    |             |     |                 |        | leave the      | emoticons     | in,      | there | is a negative     | impact | on the |
1.2 CharacteristicsofTweets
|         |          |      |      |                    |     |            | accuracies        | of the | MaxEnt                             | and | SVM classifiers, | but little | ef- |
| ------- | -------- | ---- | ---- | ------------------ | --- | ---------- | ----------------- | ------ | ---------------------------------- | --- | ---------------- | ---------- | --- |
| Twitter | messages | have | many | unique attributes, |     | which dif- |                   |        |                                    |     |                  |            |     |
|         |          |      |      |                    |     |            | fectonNaiveBayes. |        | Thedifferenceliesinthemathematical |     |                  |            |     |
ferentiates our research from previous research: models and feature weight selection of MaxEnt and SVM.
Length The maximum length of a Twitter message is 140 Stripping out the emoticons causes the classifier to learn
characters. From our training set, we calculate that the fromtheotherfeatures(e.g. unigramsandbigrams)present
averagelengthofatweetis14wordsor78characters. This inthetweet. Theclassifierusesthesenon-emoticonfeatures
is very different from the previous sentiment classification todeterminethesentiment. Thisisaninterestingside-effect
research that focused on classifying longer bodies of work, of our approach. If the test data contains an emoticon, it
such as movie reviews. does not influence the classifier because emoticon features
|                   |     |         |            |     |               |     | arenotpartofitstrainingdata. |     |     |     | Thisisacurrentlimitation |     |     |
| ----------------- | --- | ------- | ---------- | --- | ------------- | --- | ---------------------------- | --- | --- | --- | ------------------------ | --- | --- |
| Data availability |     | Another | difference | is  | the magnitude | of  |                              |     |     |     |                          |     |     |
ofourapproachbecauseitwouldbeusefultotakeemoticons
| data available.                     |     | With the | Twitter | API,                 | it is very | easy to |              |      |             |     |            |     |     |
| ----------------------------------- | --- | -------- | ------- | -------------------- | ---------- | ------- | ------------ | ---- | ----------- | --- | ---------- | --- | --- |
|                                     |     |          |         |                      |            |         | into account | when | classifying |     | test data. |     |     |
| collectmillionsoftweetsfortraining. |     |          |         | Inpastresearch,tests |            |         |              |      |             |     |            |     |     |
only consisted of thousands of training items. We consider emoticons as noisy labels because they are not
|     |     |     |     |     |     |     | perfectatdefiningthecorrectsentimentofatweet. |     |     |     |     | Thiscan |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------------------- | --- | --- | --- | --- | ------- | --- |
Language model Twitter users post messages from many beseeninthefollowingtweet: @BATMANNN:(ilovechut-
different media, including their cell phones. The frequency ney....... Without the emoticon, most people would prob-
of misspellings and slang in tweets is much higher than in ably consider this tweet to be positive. Tweets with these
other domains. types of mismatched emoticons are used to train our classi-
fiersbecausetheyaredifficulttofilteroutfromourtraining
DomainTwitteruserspostshortmessagesaboutavariety
data.
| of topics | unlike | other sites | which | are tailored |     | to a specific |     |     |     |     |     |     |     |
| --------- | ------ | ----------- | ----- | ------------ | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
topic. This differs from a large percentage of past research, 2.3 FeatureReduction
| which focused | on  | specific | domains | such as | movie | reviews. |             |          |     |       |                 |             |     |
| ------------- | --- | -------- | ------- | ------- | ----- | -------- | ----------- | -------- | --- | ----- | --------------- | ----------- | --- |
|               |     |          |         |         |       |          | The Twitter | language |     | model | has many unique | properties. |     |
Wetakeadvantageofthefollowingpropertiestoreducethe
2. APPROACH
feature space.
Ourapproachistousedifferentmachinelearningclassifiers
and feature extractors. The machine learning classifiers are UsernamesUsersoftenincludeTwitterusernamesintheir
Naive Bayes, Maximum Entropy (MaxEnt), and Support tweets in order to direct their messages. A de facto stan-
Vector Machines (SVM). The feature extractors are uni- dard is to include the @ symbol before the username (e.g.
| grams, bigrams,                                   |                                          | unigrams                           | and | bigrams, | and unigrams | with |                                                      |       |             |            |                  |     |     |
| ------------------------------------------------- | ---------------------------------------- | ---------------------------------- | --- | -------- | ------------ | ---- | ---------------------------------------------------- | ----- | ----------- | ---------- | ---------------- | --- | --- |
|                                                   |                                          |                                    |     |          |              |      | @alecmgo).                                           | An    | equivalence | class      | token (USERNAME) |     | re- |
| partofspeechtags.                                 |                                          | Webuildaframeworkthattreatsclassi- |     |          |              |      |                                                      |       |             |            |                  |     |     |
|                                                   |                                          |                                    |     |          |              |      | places all                                           | words | that        | start with | the @ symbol.    |     |     |
| fiersandfeatureextractorsastwodistinctcomponents. |                                          |                                    |     |          |              | This |                                                      |       |             |            |                  |     |     |
| 1The                                              |                                          |                                    |     |          |              |      | UsageoflinksUsersveryoftenincludelinksintheirtweets. |       |             |            |                  |     |     |
| URL                                               | is http://twittersentiment.appspot.com/. |                                    |     |          |              | This |                                                      |       |             |            |                  |     |     |
pagehasalinktoourtrainingdataandtestdata. Itisalso An equivalence class is used for all URLs. That is, we con-
a public tool that other researchers can use to build their vert a URL like“http://tinyurl.com/cvvg9a”to the token
| own data | sets. |     |     |     |     |     | “URL.” |     |     |     |     |     |     |
| -------- | ----- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |

Table 1: Example Tweets
Sentiment Query Tweet
Positive jquery dcostalis: Jquery is my new best friend.
Neutral San Francisco schuyler: just landed at San Francisco
Negative exam jvici0us: History exam studying ugh.
3.3 MaximumEntropy
Table 2: Effect of Feature Reduction
TheideabehindMaximumEntropymodelsisthatoneshould
Feature Reduction # of Features Percent of Original
preferthemostuniformmodelsthatsatifyagivenconstraint
None 794876 100.00%
[7]. MaxEnt models are feature-based models. In a two-
Username 449714 56.58%
class scenario, it is the same as using logistic regression to
URLs 730152 91.86%
findadistributionovertheclasses. MaxEntmakesnoinde-
Repeated Letters 773691 97.33%
pendence assumptions for its features, unlike Naive Bayes.
All 364464 45.85%
Thismeanswecanaddfeatureslikebigramsandphrasesto
MaxEnt without worrying about features overlapping. The
model is represented by the following:
Repeated letters Tweets contain very casual language.
Forexample,ifyousearch“hungry”withanarbitrarynum-
ber of u’s in the middle (e.g. huuuungry, huuuuuuungry,
huuuuuuuuuungry) on Twitter, there will most likely be a P (c|d,λ)= exp[Σ i λ i f i (c,d)]
nonempty result set. We use preprocessing so that any let- ME Σ c(cid:48)exp[Σ i λ i f i (c,d)]
teroccurringmorethantwotimesinarowisreplacedwith
two occurrences. In the samples above, these words would
In this formula, c is the class, d is the tweet, and λ is a
be converted into the token huungry.
weightvector. Theweightvectorsdecidethesignificanceof
a feature in classification. A higher weight means that the
Table 2 shows the effect of these feature reductions. These
featureisastrongindicatorfortheclass. Theweightvector
three reductions shrink the feature set down to 45.85% of
is found by numerical optimization of the lambdas so as to
its original size.
maximize the conditional probability.
3. MACHINELEARNINGMETHODS WeusetheStanfordClassifier3 toperformMaxEntclassifi-
We test different classifiers: keyword-based, Naive Bayes, cation. Fortrainingtheweightsweusedconjugategradient
maximum entropy, and support vector machines. ascent and added smoothing (L2 regularization).
3.1 Baseline Theoretically,MaxEntperformsbetterthanNaiveBayesbe-
causeithandlesfeatureoverlapbetter. However,inpractice,
Twittratr is a website that performs sentiment analysis on
Naive Bayes can still perform well on a variety of problems
tweets. Their approach is to use a list of positive and neg-
[7].
ative keywords. As a baseline, we use Twittratr’s list of
keywords, which is publicly available2. This list consists of
174positivewordsand185negativewords. Foreachtweet, 3.4 SupportVectorMachines
wecountthenumberofnegativekeywordsandpositivekey- Support Vector Machines is another popular classification
words that appear. This classifier returns the polarity with technique [2]. We use the SVMlight [4] software with a
thehighercount. Ifthereisatie,thenpositivepolarity(the linear kernel. Our input data are two sets of vectors of size
majority class) is returned. m. Each entry in the vector corresponds to the presence
a feature. For example, with a unigram feature extractor,
3.2 NaiveBayes eachfeatureisasinglewordfoundinatweet. Ifthefeature
Naive Bayes is a simple model which works well on text is present, the value is 1, but if the feature is absent, then
categorization[5]. WeuseamultinomialNaiveBayesmodel. the value is 0. We use feature presence, as opposed to a
Class c∗ is assigned to tweet d, where count,sothatwedonothavetoscaletheinputdata,which
speeds up overall processing [1].
c∗=argmac P (c|d)
c NB
(cid:80) 4. EVALUATION
(P(c) m P(f|c)ni(d))
P (c|d):= i=1 4.1 ExperimentalSet-up
NB P(d)
There are not any large public data sets of Twitter mes-
In this formula, f represents a feature and n i (d) represents sages with sentiment, so we collect our own data. Twitter
the count of feature f i found in tweet d. There are a total has an Application Programming Interface (API)4 for pro-
of m features. Parameters P(c) and P(f|c) are obtained grammaticallyaccessingtweetsbyqueryterm. TheTwitter
through maximum likelihood estimates, and add-1 smooth-
3The Stanford Classifier can be downloaded from
ing is utilized for unseen features.
http://nlp.stanford.edu/software/classifier.shtml.
2The list of keywords is linked off of http://twitrratr.com/. 4More information about the Twitter API can be found at
We have no association with Twittratr. http://apiwiki.twitter.com/.

|           | Table       | 3: List | of           | Emoticons |          |        |               |     |              |          |        |          |
| --------- | ----------- | ------- | ------------ | --------- | -------- | ------ | ------------- | --- | ------------ | -------- | ------ | -------- |
| Emoticons | mapped      | to      | :) Emoticons |           | mapped   | to :(  |               |     |              |          |        |          |
|           | :)          |         |              |           | :(       |        |               |     |              |          |        |          |
|           |             |         |              |           |          |        | Table 4: List | of  | Queries Used | to       | Create | Test Set |
|           | :-)         |         |              |           | :-(      |        |               |     |              |          |        |          |
|           |             |         |              |           |          |        | Query         |     | Negative     | Positive | Total  | Category |
|           | : )         |         |              |           | : (      |        |               |     |              |          |        |          |
|           |             |         |              |           |          |        | 40d           |     |              | 2        | 2      | Product  |
|           | :D          |         |              |           |          |        | 50d           |     |              | 5        | 5      | Product  |
|           | =)          |         |              |           |          |        | aig           |     | 7            |          | 7      | Company  |
|           |             |         |              |           |          |        | at&t          |     | 13           |          | 13     | Company  |
|           |             |         |              |           |          |        | bailout       |     | 1            |          | 1      | Misc.    |
|           |             |         |              |           |          |        | bing          |     | 1            |          | 1      | Product  |
|           |             |         |              |           |          |        | BobbyFlay     |     |              | 6        | 6      | Person   |
| API has   | a parameter | that    | specifies    | which     | language | to re- |               |     |              |          |        |          |
|           |             |         |              |           |          |        | boozallen     |     | 1            | 2        | 3      | Company  |
trieve tweets in. We always set this parameter to English. carwarrantycall 2 2 Misc.
Thus, our classification will only work on tweets in English cheney 5 5 Person
because our training data is English-only. comcast 4 4 Company
|     |     |     |     |     |     |     | DannyGokey |     |     | 4   | 4   | Person |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | ------ |
|     |     |     |     |     |     |     | dentist    |     | 9   | 3   | 12  | Misc.  |
Therearemultipleemoticonsthatcanexpresspositiveemo- eastpaloalto 1 2 3 Location
tion and negative emotion. For example, :) and :-) both espn 1 1 Product
expresspositiveemotion. IntheTwitterAPI,thequery“:)” exam 5 2 7 Misc.
|             |        |              |          |     |            |         | federer    |     |     | 1   | 1   | Person |
| ----------- | ------ | ------------ | -------- | --- | ---------- | ------- | ---------- | --- | --- | --- | --- | ------ |
| will return | tweets | that contain | positive |     | emoticons, | and the |            |     |     |     |     |        |
|             |        |              |          |     |            |         | fredwilson |     |     | 2   | 2   | Person |
query“:(”will return tweets with negative emoticons5. The g2 7 7 Product
full list of emoticons can be found in Table 3. gm 16 16 Company
|     |     |     |     |     |     |     | goodbysilverstein |     |     | 6   | 6   | Company |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- | ------- |
|     |     |     |     |     |     |     | google            |     | 1   | 4   | 5   | Company |
For the training data, we use a scraper that queries the googleio 4 4 Event
|         |                    |     |             |     |         |              | indiaelection  |     |     | 1   | 1   | Event |
| ------- | ------------------ | --- | ----------- | --- | ------- | ------------ | -------------- | --- | --- | --- | --- | ----- |
| Twitter | API. Periodically, |     | the scraper |     | sends a | query for :) |                |     |     |     |     |       |
|         |                    |     |             |     |         |              | indianelection |     |     | 1   | 1   | Event |
andaseparatequeryfor:(atthesametime. Thisallowsus insects 5 1 6 Misc.
to collect tweets that contain the emoticons listed in Table iphoneapp 1 1 2 Product
| 3.  |     |     |     |     |     |     | iran   |     | 4   |     | 4   | Location |
| --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | -------- |
|     |     |     |     |     |     |     | itchy  |     | 5   |     | 5   | Misc.    |
|     |     |     |     |     |     |     | jquery |     | 1   | 3   | 4   | Product  |
TheTwitterAPIhasalimitof100tweetsinaresponsefor jquerybook 2 2 Product
any request. The scraper has a parameter that allows us to kindle2 1 16 17 Product
specify the frequency of polling. We found an interval of 2 lakers 4 4 Product
|              |                 |         |          |            |         |           | lambdacalculus  |     | 2   | 1   | 3   | Misc.   |
| ------------ | --------------- | ------- | -------- | ---------- | ------- | --------- | --------------- | --- | --- | --- | --- | ------- |
| minutes      | is a reasonable | polling |          | parameter. | The     | tweets in |                 |     |     |     |     |         |
|              |                 |         |          |            |         |           | latex           |     | 5   | 3   | 8   | Misc.   |
|              |                 |         |          |            |         |           | lebron          |     | 4   | 14  | 18  | Person  |
| our training | set are         | from    | the time | period     | between | April 6,  |                 |     |     |     |     |         |
|              |                 |         |          |            |         |           | lyx             |     |     | 2   | 2   | Misc.   |
| 2009 to      | June 25, 2009.  |         |          |            |         |           |                 |     |     |     |     |         |
|              |                 |         |          |            |         |           | MalcolmGladwell |     | 3   | 7   | 10  | Person  |
|              |                 |         |          |            |         |           | mashable        |     |     | 2   | 2   | Product |
Thetrainingdataispost-processedwiththefollowingfilters: mcdonalds 1 5 6 Company
|     |     |     |     |     |     |     | naivebayes       |     | 1   |     | 1   | Misc.   |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | --- | --- | --- | ------- |
|     |     |     |     |     |     |     | nightatthemuseum |     | 3   | 12  | 15  | Movie   |
|     |     |     |     |     |     |     | nike             |     | 4   | 11  | 15  | Company |
1. Emoticons listed in Table 3 are stripped off. This is northkorea 6 6 Location
important for training purposes. If the emoticons are notredameschool 2 2 Misc.
|     |     |     |     |     |     |     | obama |     | 1   | 9   | 10  | Person |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | ------ |
notstrippedoff,thentheMaxEntandSVMclassifiers
|     |     |     |     |     |     |     | pelosi |     | 4   |     | 4   | Person |
| --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | ------ |
tendtoputalargeamountofweightontheemoticons,
|       |       |           |     |     |     |     | republican   |     | 1   |     | 1   | Misc.    |
| ----- | ----- | --------- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | -------- |
| which | hurts | accuracy. |     |     |     |     | safeway      |     | 5   | 2   | 7   | Company  |
|       |       |           |     |     |     |     | sanfrancisco |     | 3   | 1   | 4   | Location |
|       |       |           |     |     |     |     | scrapbooking |     |     | 1   | 1   | Misc.    |
2. Anytweetcontainingbothpositiveandnegativeemoti-
|     |     |     |     |     |     |     | shorelineamphitheatre |     |     | 1   | 1   | Location |
| --- | --- | --- | --- | --- | --- | --- | --------------------- | --- | --- | --- | --- | -------- |
consareremoved. Thismayhappenifatweetcontains sleep 3 1 4 Misc.
two subjects. Here is an example of a tweet with this stanford 7 7 Misc.
property: Target orientation :( But it is my birthday startrek 4 4 Movie
|       |                                         |     |     |     |     |     | summize |     | 2   |     | 2   | Product |
| ----- | --------------------------------------- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | ------- |
| today | :). Thesetweetsareremovedbecausewedonot |     |     |     |     |     |         |     |     |     |     |         |
|       |                                         |     |     |     |     |     | surgery |     | 1   |     | 1   | Misc.   |
want positive features marked as part of a negative timewarner 33 33 Company
|             |             |          |            |        |         |            | twitter        |     |     | 1   | 1   | Company |
| ----------- | ----------- | -------- | ---------- | ------ | ------- | ---------- | -------------- | --- | --- | --- | --- | ------- |
| tweet,      | or negative | features |            | marked | as part | of a posi- |                |     |     |     |     |         |
|             |             |          |            |        |         |            | twitterapi     |     | 6   | 2   | 8   | Product |
| tive        | tweet.      |          |            |        |         |            | viralmarketing |     | 1   | 2   | 3   | Misc.   |
|             |             |          |            |        |         |            | visa           |     |     | 1   | 1   | Company |
|             |             |          |            |        |         |            | visacard       |     | 1   |     | 1   | Product |
| 3. Retweets | are         | removed. | Retweeting |        | is the  | process of |                |     |     |     |     |         |
|             |             |          |            |        |         |            | warrenbuffet   |     |     | 5   | 5   | Person  |
| copying     | another     | user’s   | tweet      | and    | posting | to another |                |     |     |     |     |         |
|             |             |          |            |        |         |            | waves&box      |     |     | 1   | 1   | Product |
account. This usually happens if a user likes another weka 1 1 Product
user’stweet. Retweetsarecommonlyabbreviatedwith wieden 1 1 Company
|                                            |                    |     |     |       |        |             | wolframalpha |     | 1   | 2   | 3   | Product |
| ------------------------------------------ | ------------------ | --- | --- | ----- | ------ | ----------- | ------------ | --- | --- | --- | --- | ------- |
| “RT.”Forexample,considerthefollowingtweet: |                    |     |     |       |        | Awe-        |              |     |     |     |     |         |
|                                            |                    |     |     |       |        |             | worldcup     |     |     | 1   | 1   | Event   |
| some!                                      | RT @rupertgrintnet |     |     | Harry | Potter | Marks Place |              |     |     |     |     |         |
|                                            |                    |     |     |       |        |             | worldcup2010 |     |     | 1   | 1   | Event   |
in Film History http://bit.ly/Eusxi :). In this case, yahoo 1 1 Company
|                     |              |                                  |       |         |             |            | yankees |     |     | 1   | 1   | Misc. |
| ------------------- | ------------ | -------------------------------- | ----- | ------- | ----------- | ---------- | ------- | --- | --- | --- | --- | ----- |
| 5At                 |              |                                  |       |         |             |            | Total   |     | 177 | 182 | 359 | -     |
| the                 | time of this | writing,                         | the   | Twitter | API         | query “:(” |         |     |     |     |     |       |
| returns messages    | with“:P”,    |                                  | which | does    | not usually | express    |         |     |     |     |     |       |
| anegativesentiment. |              | Messageswith:Parefilteredoutfrom |       |         |             |            |         |     |     |     |     |       |
| our training        | data.        |                                  |       |         |             |            |         |     |     |     |     |       |

|     |     |     |     |     |     |     |     | MaxEnt, | and SVM, | respectively. |     | This | is very | similar | to  |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | -------- | ------------- | --- | ---- | ------- | ------- | --- |
Table 5: Categories for Test Data our results of 81.3%, 80.5%, and 82.2% for the same set of
|     | Category |     | Total | Percent |     |     |     |     |     |     |     |     |     |     |     |
| --- | -------- | --- | ----- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
classifiers.
|     | Company  |     | 119 | 33.15% |     |     |     |                  |           |          |             |           |           |          |         |
| --- | -------- | --- | --- | ------ | --- | --- | --- | ---------------- | --------- | -------- | ----------- | --------- | --------- | -------- | ------- |
|     | Event    |     | 8   | 2.23%  |     |     |     |                  |           |          |             |           |           |          |         |
|     |          |     |     |        |     |     |     | Bigrams          | We use    | bigrams  | to          | help with | tweets    | that     | contain |
|     | Location |     | 18  | 5.01%  |     |     |     |                  |           |          |             |           |           |          |         |
|     |          |     |     |        |     |     |     | negated          | phrases   | like“not | good”or“not |           | bad.”In   | our      | exper-  |
|     | Misc.    |     | 67  | 18.66% |     |     |     |                  |           |          |             |           |           |          |         |
|     |          |     |     |        |     |     |     | iments, negation |           | as an    | explicit    | feature   | with      | unigrams | does    |
|     | Movie    |     | 19  | 5.29%  |     |     |     |                  |           |          |             |           |           |          |         |
|     |          |     |     |        |     |     |     | not improve      | accuracy, |          | so we       | are very  | motivated | to       | try bi- |
|     | Person   |     | 65  | 18.11% |     |     |     |                  |           |          |             |           |           |          |         |
grams.
|     | Product |     | 63  | 17.55% |     |     |     |     |     |     |     |     |     |     |     |
| --- | ------- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Grand Total 359 However,bigramstendtobeverysparseandtheoverallac-
|                                     |         |                |              |                  |              |             |          | curacy drops  | in             | the case     | of both              | MaxEnt         | and       | SVM.       | Even     |
| ----------------------------------- | ------- | -------------- | ------------ | ---------------- | ------------ | ----------- | -------- | ------------- | -------------- | ------------ | -------------------- | -------------- | --------- | ---------- | -------- |
|                                     |         |                |              |                  |              |             |          | collapsing    | the individual |              | words                | to equivalence |           | classes    | does     |
| the                                 | user is | rebroadcasting |              | rupertgrintnet’s |              | tweet       | and      |               |                |              |                      |                |           |            |          |
|                                     |         |                |              |                  |              |             |          | not help.     | The problem    |              | of sparseness        | can            | be        | seen in    | the fol- |
| adding                              | the     | comment        | Awesome!.    |                  | Any          | tweet       | with RT  |               |                |              |                      |                |           |            |          |
|                                     |         |                |              |                  |              |             |          | lowing tweet: | @stellargirl   |              | I loooooooovvvvvveee |                |           | my         | Kindle2. |
| is removed                          |         | from the       | training     | data             | to           | avoid       | giving a |               |                |              |                      |                |           |            |          |
|                                     |         |                |              |                  |              |             |          | Not that      | the DX         | is cool,     | but                  | the 2 is       | fantastic | in         | its own  |
| particular                          | tweet   | extra          | weight       | in               | the training | data.       |          |               |                |              |                      |                |           |            |          |
|                                     |         |                |              |                  |              |             |          | right. MaxEnt | gave           | equal        | probabilities        |                | to the    | positive   | and      |
|                                     |         |                |              |                  |              |             |          | negative      | class for      | this         | case because         | there          | is        | not a      | bigram   |
| 4. Tweets                           | with    | “:P”           | are removed. |                  | At the       | time        | of this  |               |                |              |                      |                |           |            |          |
|                                     |         |                |              |                  |              |             |          | that tips     | the polarity   | in           | either               | direction.     |           |            |          |
| writing,                            | the     | Twitter        | API          | has an           | issue in     | which       | tweets   |               |                |              |                      |                |           |            |          |
| with“:P”arereturnedforthequery“:(”. |         |                |              |                  |              | Thesetweets |          |               |                |              |                      |                |           |            |          |
|                                     |         |                |              |                  |              |             |          | In general    | using          | only bigrams |                      | as features    | is        | not useful | be-      |
areremovedbecause“:P”usuallydoesnotimplyaneg-
|                              |            |     |     |                       |     |     |     | causethefeaturespaceisverysparse. |             |     |              | Itisbettertocombine |     |     |     |
| ---------------------------- | ---------- | --- | --- | --------------------- | --- | --- | --- | --------------------------------- | ----------- | --- | ------------ | ------------------- | --- | --- | --- |
| ative                        | sentiment. |     |     |                       |     |     |     |                                   |             |     |              |                     |     |     |     |
|                              |            |     |     |                       |     |     |     | unigrams                          | and bigrams |     | as features. |                     |     |     |     |
| 5. Repeatedtweetsareremoved. |            |     |     | Occasionally,theTwit- |     |     |     |                                   |             |     |              |                     |     |     |     |
ter API returns duplicate tweets. The scraper com- Unigrams and Bigrams Both unigrams and bigrams are
paresatweettothelast100tweets. Ifitmatchesany, used as features. Compared to unigram features, accuracy
then it discards the tweet. Similar to retweets, dupli- improved for Naive Bayes (81.3% from to82.7%)andMax-
catesareremovedtoavoidputtingextraweightonany Ent (from 80.5 to 82.7). However, there was a decline for
particular tweet. SVM (from 82.2% to 81.6%). For Pang and Lee, there was
adeclineforNaiveBayesandSVM,butanimprovementfor
MaxEnt.
| After post-processing |     | the | data, | we take | the | first | 800,000 |     |     |     |     |     |     |     |     |
| --------------------- | --- | --- | ----- | ------- | --- | ----- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
tweetswithpositiveemoticons,and800,000tweetswithneg- Parts of speech We use part of speech (POS) tags as fea-
ative emoticons, for a total of 1,600,000 training tweets. turesbecausethesamewordmayhavemanydifferentmean-
|                                              |           |                |                                |           |               |          |          | ings depending                      | on          | its usage.   |          | For example,“over”as |         |               | a verb  |
| -------------------------------------------- | --------- | -------------- | ------------------------------ | --------- | ------------- | -------- | -------- | ----------------------------------- | ----------- | ------------ | -------- | -------------------- | ------- | ------------- | ------- |
| The test data                                | is        | manually       | collected,                     | using     | the           | web      | applica- |                                     |             |              |          |                      |         |               |         |
|                                              |           |                |                                |           |               |          |          | may have                            | a negative  | connotation. |          | “Over”may            |         | also          | be used |
| tion. A set                                  | of 177    | negative       | tweets                         | and       | 182           | positive | tweets   |                                     |             |              |          |                      |         |               |         |
|                                              |           |                |                                |           |               |          |          | as a noun                           | to refer    | to the       | cricket  | over,                | which   | does not      | carry   |
| weremanuallymarked.                          |           |                | Notallthetestdatahasemoticons. |           |               |          |          |                                     |             |              |          |                      |         |               |         |
|                                              |           |                |                                |           |               |          |          | a positive                          | or negative | connotation. |          |                      |         |               |         |
| We use the                                   | following | process        | to                             | collect   | test data:    |          |          |                                     |             |              |          |                      |         |               |         |
|                                              |           |                |                                |           |               |          |          | WefoundthatthePOStagswerenotuseful. |             |              |          |                      |         | Thisisconsis- |         |
|                                              |           |                |                                |           |               |          |          | tent with                           | Pang        | and Lee      | [9]. The | accuracy             | for     | Naive         | Bayes   |
| 1. WesearchtheTwitterAPIwithspecificqueries. |           |                |                                |           |               |          | These    |                                     |             |              |          |                      |         |               |         |
|                                              |           |                |                                |           |               |          |          | and SVM                             | decreased   | while        | the      | performance          | for     | MaxEnt        | in-     |
| queries                                      | are       | arbitrarily    | chosen                         | from      | different     | domains. |          |                                     |             |              |          |                      |         |               |         |
|                                              |           |                |                                |           |               |          |          | creased negligibly                  |             | when         | compared | to the               | unigram | results.      |         |
| For                                          | example,  | these          | queries                        | consist   | of consumer   |          | prod-    |                                     |             |              |          |                      |         |               |         |
| ucts                                         | (40d,     | 50d, kindle2), |                                | companies | (aig,         | at&t),   | and      |                                     |             |              |          |                      |         |               |         |
| people(BobbyFlay,WarrenBuffet).              |           |                |                                |           | Thequeryterms |          |          | 5. FUTUREWORK                       |             |              |          |                      |         |               |         |
we used are listed in Table 4. The different categories Machinelearningtechniquesperformwellforclassifyingsen-
of these queries are listed in Table 5. timent in tweets. We believe that the accuracy could still
|                                   |          |              |      |                 |                |               |     | be improved.    | Below | is  | a list of | ideas we | think | could | help in |
| --------------------------------- | -------- | ------------ | ---- | --------------- | -------------- | ------------- | --- | --------------- | ----- | --- | --------- | -------- | ----- | ----- | ------- |
| 2. Welookattheresultsetforaquery. |          |              |      |                 | Ifweseearesult |               |     |                 |       |     |           |          |       |       |         |
|                                   |          |              |      |                 |                |               |     | this direction. |       |     |           |          |       |       |         |
| that                              | contains | a sentiment, |      | we mark         | it             | as positive   | or  |                 |       |     |           |          |       |       |         |
| negative.                         | Thus,    | this         | test | set is selected |                | independently |     |                 |       |     |           |          |       |       |         |
SemanticsOuralgorithmsclassifytheoverallsentimentof
| of the | presence | of  | emoticons. |     |     |     |     |          |              |              |            |           |        |              |          |
| ------ | -------- | --- | ---------- | --- | --- | --- | --- | -------- | ------------ | ------------ | ---------- | --------- | ------ | ------------ | -------- |
|        |          |     |            |     |     |     |     | a tweet. | The polarity |              | of a tweet | may       | depend | on           | the per- |
|        |          |     |            |     |     |     |     | spective | you are      | interpreting |            | the tweet | from.  | For example, |          |
4.2 ResultsandDiscussion inthetweetFedererbeatsNadal:),thesentimentispositive
We explore the usage of unigrams, bigrams, unigrams and for Federer and negative for Nadal. In this case, semantics
bigrams, and parts of speech as features. Table 6 summa- mayhelp. Usingasemanticrolelabelermayindicatewhich
rizes the results. noun is mainly associated with the verb and the classifi-
|     |     |     |     |     |     |     |     | cation would | take | place | accordingly. | This | may | allow | Nadal |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ---- | ----- | ------------ | ---- | --- | ----- | ----- |
Unigrams The unigram feature extractor is the simplest beatsFederer:) tobeclassifieddifferentlyfromFedererbeats
| waytoretrievefeaturesfromatweet. |     |     |     |     | Themachinelearning |     |     | Nadal :). |     |     |     |     |     |     |     |
| -------------------------------- | --- | --- | --- | --- | ------------------ | --- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
algorithmsclearlyperformbetterthanourkeywordbaseline.
These results are very similar to Pang and Lee [9]. They Domain-specific tweets Our best classifier has an accu-
report 81.0%, 80.4%, and 82.9% accuracy for Naive Bayes, racy of 83.0% for tweets across all domains. This is a very

|     |     |     |     |          |     | Table  | 6: Classifier | Accuracy    |        |     |      |     |     |     |     |
| --- | --- | --- | --- | -------- | --- | ------ | ------------- | ----------- | ------ | --- | ---- | --- | --- | --- | --- |
|     |     |     |     | Features |     |        | Keyword       | Naive Bayes | MaxEnt |     | SVM  |     |     |     |     |
|     |     |     |     | Unigram  |     |        | 65.2          | 81.3        | 80.5   |     | 82.2 |     |     |     |     |
|     |     |     |     | Bigram   |     |        | N/A           | 81.6        | 79.1   |     | 78.8 |     |     |     |     |
|     |     |     |     | Unigram  | +   | Bigram | N/A           | 82.7        | 83.0   |     | 81.6 |     |     |     |     |
|     |     |     |     | Unigram  | +   | POS    | N/A           | 79.9        | 79.9   |     | 81.9 |     |     |     |     |
large vocabulary. If limited to particular domains (such as TheauthorswouldliketothankChristopherManning,Nate
movies) we feel our classifiers may perform better. Chambers, and Abhay Shete for providing feedback on this
research.
| Handling | neutral | tweetsInrealworldapplications,neu- |     |     |     |     |     |     |     |     |     |     |     |     |     |
| -------- | ------- | ---------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
traltweetscannotbeignored. Properattentionneedstobe 9. REFERENCES
| paid to | neutral | sentiment. |     |     |     |     |     |           |           |             |        |         |         |        |     |
| ------- | ------- | ---------- | --- | --- | --- | --- | --- | --------- | --------- | ----------- | ------ | ------- | ------- | ------ | --- |
|         |         |            |     |     |     |     |     | [1] D. O. | Computer, |             | C. wei | Hsu, C. | chung   | Chang, | and |
|         |         |            |     |     |     |     |     | C. jen    | Lin.      | A practical | guide  | to      | support | vector |     |
Internationalization WefocusonlyonEnglishsentences, classification chih-wei hsu, chih-chung chang, and
butTwitterhasmanyinternationalusers. Itshouldbepos- chih-jen lin. Technical report, 2003.
| sible to | use our | approach | to  | classify | sentiment | in  | other lan- |                                       |     |     |     |     |     |              |     |
| -------- | ------- | -------- | --- | -------- | --------- | --- | ---------- | ------------------------------------- | --- | --- | --- | --- | --- | ------------ | --- |
|          |         |          |     |          |           |     |            | [2] N.CristianiniandJ.Shawe-Taylor.An |     |     |     |     |     | Introduction | to  |
guages.
|           |          |     |         |     |          |           |     | Support  | Vector   | Machines |           | and Other  | Kernel-based |        |       |
| --------- | -------- | --- | ------- | --- | -------- | --------- | --- | -------- | -------- | -------- | --------- | ---------- | ------------ | ------ | ----- |
|           |          |     |         |     |          |           |     | Learning | Methods. |          | Cambridge | University |              | Press, | March |
| Utilizing | emoticon |     | data in | the | test set | Emoticons | are |          |          |          |           |            |              |        |       |
2000.
| strippedfromourtrainingdata. |          |             |          | Thismeansthatifourtest |           |              |           |                |         |             |           |            |                    |              |     |
| ---------------------------- | -------- | ----------- | -------- | ---------------------- | --------- | ------------ | --------- | -------------- | ------- | ----------- | --------- | ---------- | ------------------ | ------------ | --- |
|                              |          |             |          |                        |           |              |           | [3] B. J.      | Jansen, | M. Zhang,   |           | K. Sobel,  | and                | A. Chowdury. |     |
| data contains                |          | an emoticon | feature, |                        | this does | not          | influence |                |         |             |           |            |                    |              |     |
|                              |          |             |          |                        |           |              |           | Micro-blogging |         | as          | online    | word of    | mouth              | branding.    | In  |
| the classifier               | towards  | a           | class.   | This                   | should    | be addressed | be-       |                |         |             |           |            |                    |              |     |
|                              |          |             |          |                        |           |              |           | CHI            | EA ’09: | Proceedings |           | of the     | 27th international |              |     |
| cause the                    | emoticon | features    |          | are very               | valuable. |              |           |                |         |             |           |            |                    |              |     |
|                              |          |             |          |                        |           |              |           | conference     |         | extended    | abstracts | on         | Human              | factors      | in  |
|                              |          |             |          |                        |           |              |           | computing      |         | systems,    | pages     | 3859–3864, | New                | York,        | NY, |
6. RELATEDWORK
|                        |          |            |                             |        |                  |             |           | USA,             | 2009.      | ACM.   |             |            |             |          |        |
| ---------------------- | -------- | ---------- | --------------------------- | ------ | ---------------- | ----------- | --------- | ---------------- | ---------- | ------ | ----------- | ---------- | ----------- | -------- | ------ |
| There has              | been     | a large    | amount                      | of     | prior            | research    | in senti- |                  |            |        |             |            |             |          |        |
|                        |          |            |                             |        |                  |             |           | [4] T. Joachims. |            | Making | large-scale |            | support     | vector   |        |
| ment analysis,         |          | especially | in the                      | domain | of               | product     | reviews,  |                  |            |        |             |            |             |          |        |
|                        |          |            |                             |        |                  |             |           | machine          | learning   |        | practical.  | In B.      | Sch¨olkopf, | C.       | J. C.  |
| moviereviews,andblogs. |          |            | PangandLee[8]isanup-to-date |        |                  |             |           |                  |            |        |             |            |             |          |        |
|                        |          |            |                             |        |                  |             |           | Burges,          | and        | A. J.  | Smola,      | editors,   | Advances    | in       | kernel |
| survey of              | previous | work       | in sentiment                |        | analysis.        | Researchers |           |                  |            |        |             |            |             |          |        |
|                        |          |            |                             |        |                  |             |           | methods:         | support    |        | vector      | learning,  | pages       | 169–184. | MIT    |
| have also              | analyzed | the        | brand                       | impact | of microblogging |             | [3].      |                  |            |        |             |            |             |          |        |
|                        |          |            |                             |        |                  |             |           | Press,           | Cambridge, |        | MA,         | USA, 1999. |             |          |        |
We could not find any papers that use machine learning [5] C. D. Manning and H. Schutze. Foundations of
| techniques | in    | the specific | domain |        | of microblogs, |      | probably  |             |         |     |          |             |     |        |     |
| ---------- | ----- | ------------ | ------ | ------ | -------------- | ---- | --------- | ----------- | ------- | --- | -------- | ----------- | --- | ------ | --- |
|            |       |              |        |        |                |      |           | statistical | natural |     | language | processing. | MIT | Press, |     |
| because    | these | services     | have   | become | popular        | only | in recent |             |         |     |          |             |     |        |     |
1999.
years.
|                     |     |       |         |          |     |           |         | [6] G. Mishne. |                 | Experiments |          | with mood    | classification |          | in  |
| ------------------- | --- | ----- | ------- | -------- | --- | --------- | ------- | -------------- | --------------- | ----------- | -------- | ------------ | -------------- | -------- | --- |
|                     |     |       |         |          |     |           |         | blog           | posts.          | In 1st      | Workshop | on Stylistic |                | Analysis | Of  |
| Text classification |     | using | machine | learning |     | is a well | studied |                |                 |             |          |              |                |          |     |
|                     |     |       |         |          |     |           |         | Text           | For Information |             | Access,  | 2005.        |                |          |     |
field[5]. PangandLee[9]researchedtheperformanceofvar-
|              |          |     |            |        |     |        |         | [7] K. Nigam, |     | J. Lafferty, | and | A. Mccallum. |     | Using |     |
| ------------ | -------- | --- | ---------- | ------ | --- | ------ | ------- | ------------- | --- | ------------ | --- | ------------ | --- | ----- | --- |
| ious machine | learning |     | techniques | (Naive |     | Bayes, | maximum |               |     |              |     |              |     |       |     |
entropy, and support vector machines) in the specific do- maximum entropy for text classification. In IJCAI-99
main of movie reviews. We modeled much of our research Workshop on Machine Learning for Information
fromtheirresults. Theywereabletoachieveanaccuracyof Filtering, pages 61–67, 1999.
|               |       |              |                 |          |              |     |          | [8] B. Pang  | and           | L. Lee.  | Opinion | mining         | and            | sentiment |     |
| ------------- | ----- | ------------ | --------------- | -------- | ------------ | --- | -------- | ------------ | ------------- | -------- | ------- | -------------- | -------------- | --------- | --- |
| 82.9% using   | SVM   | with         | an unigram      |          | model.       |     |          |              |               |          |         |                |                |           |     |
|               |       |              |                 |          |              |     |          | analysis.    | Foundations   |          | and     | Trends         | in Information |           |     |
|               |       |              |                 |          |              |     |          | Retrieval,   | 2(1-2):1–135, |          |         | 2008.          |                |           |     |
| Read [10]     | shows | that         | using emoticons |          | as labels    | for | positive |              |               |          |         |                |                |           |     |
|               |       |              |                 |          |              |     |          | [9] B. Pang, | L.            | Lee, and | S.      | Vaithyanathan. |                | Thumbs    | up? |
| and sentiment |       | is effective | for             | reducing | dependencies |     | in ma-   |              |               |          |         |                |                |           |     |
chine learning techniques. We use the same idea for our Sentiment classification using machine learning
Twitter training data. techniques. In Proceedings of the Conference on
|                |         |                 |            |             |         |            |            | Empirical       | Methods     |                | in Natural        | Language   |              | Processing |         |
| -------------- | ------- | --------------- | ---------- | ----------- | ------- | ---------- | ---------- | --------------- | ----------- | -------------- | ----------------- | ---------- | ------------ | ---------- | ------- |
| 7. CONCLUSIONS |         |                 |            |             |         |            |            | (EMNLP),        |             | pages          | 79–86,            | 2002.      |              |            |         |
|                |         |                 |            |             |         |            |            | [10] J. Read.   | Using       | emoticons      |                   | to reduce  | dependency   |            | in      |
| We show        | that    | using           | emoticons  | as          | noisy   | labels     | for train- |                 |             |                |                   |            |              |            |         |
|                |         |                 |            |             |         |            |            | machine         | learning    |                | techniques        | for        | sentiment    |            |         |
| ing data       | is an   | effective       | way        | to perform  | distant | supervised |            |                 |             |                |                   |            |              |            |         |
|                |         |                 |            |             |         |            |            | classification. |             | In Proceedings |                   | of ACL-05, |              | 43nd       | Meeting |
| learning.      | Machine | learning        | algorithms |             | (Naive  | Bayes,     | max-       |                 |             |                |                   |            |              |            |         |
|                |         |                 |            |             |         |            |            | of the          | Association |                | for Computational |            | Linguistics. |            |         |
| imum entropy   |         | classification, |            | and support |         | vector     | machines)  |                 |             |                |                   |            |              |            |         |
can achieve high accuracy for classifying sentiment when Association for Computational Linguistics, 2005.
usingthismethod. AlthoughTwittermessageshaveunique [11] T. Wilson, J. Wiebe, and P. Hoffmann. Recognizing
characteristics compared to other corpora, machine learn- contextual polarity in phrase-level sentiment analysis.
ing algorithms are shown to classify tweet sentiment with In Proceedings of Human Language Technologies
|     |     |     |     |     |     |     |     | Conference/Conference |     |     |     | on Empirical | Methods |     | in  |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------- | --- | --- | --- | ------------ | ------- | --- | --- |
similar performance.
|     |     |     |     |     |     |     |     | Natural    | Language |           | Processing | (HLT/EMNLP |     | 2005), |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | --------- | ---------- | ---------- | --- | ------ | --- |
|     |     |     |     |     |     |     |     | Vancouver, |          | CA, 2005. |            |            |     |        |     |
8. ACKNOWLEDGMENTS

---

## Content from Previous Extraction (not in markitdown output)

### Additional Content

#### 1

## **Twitter Sentiment Classification using Distant Supervision**

#### 2

Alec Go Stanford University Stanford, CA 94305 alecmgo@stanford.edu

#### 3

Richa Bhayani Lei Huang Stanford University Stanford University Stanford, CA 94305 Stanford, CA 94305 rbhayani@stanford.edu leirocky@stanford.edu

#### 4

We introduce a novel approach for automatically classifying the sentiment of Twitter messages. These messages are classified as either positive or negative with respect to a query term. This is useful for consumers who want to research the sentiment of products before purchase, or companies that want to monitor the public sentiment of their brands. There is no previous research on classifying sentiment of messages on microblogging services like Twitter. We present the results of machine learning algorithms for classifying the sentiment of Twitter messages using distant supervision. Our training data consists of Twitter messages with emoticons, which are used as noisy labels. This type of training data is abundantly available and can be obtained through automated means. We show that machine learning algorithms (Naive Bayes, Maximum Entropy, and SVM) have accuracy above 80% when trained with emoticon data. This paper also describes the preprocessing steps needed in order to achieve high accuracy. The main contribution of this paper is the idea of using tweets with emoticons for distant supervised learning.

#### 5

## **Categories and Subject Descriptors**

#### 6

I.2 [ **Artificial Intelligence** ]: Natural Language Processing

#### 7

Twitter is a popular microblogging service where users create status messages (called “tweets”). These tweets sometimes express opinions about different topics. We propose a method to automatically extract sentiment (positive or negative) from a tweet. This is very useful because it allows feedback to be aggregated without manual intervention.

#### 8

Consumers can use sentiment analysis to research products or services before making a purchase. Marketers can use this to research public opinion of their company and products, or to analyze customer satisfaction. Organizations can also use this to gather critical feedback about problems in newly released products.

#### 9

There has been a large amount of research in the area of sentiment classification. Traditionally most of it has focused on classifying larger pieces of text, like reviews [9]. Tweets (and microblogs in general) are different from reviews primarily because of their purpose: while reviews represent summarized thoughts of authors, tweets are more casual and limited to 140 characters of text. Generally, tweets are not as thoughtfully composed as reviews. Yet, they still offer companies an additional avenue to gather feedback. There has been some work by researchers in the area of phrase level and sentence level sentiment classification recently [11]. Previous research on analyzing blog posts includes [6].

#### 10

Previous research in sentiment analysis like Pang et al. [9] have analyzed the performance of different classifiers on movie reviews. The work of Pang et al. has served as a baseline and many authors have used the techniques provided in their paper across different domains. Pang et al. also make use of a similar idea as ours, using star ratings as polarity signals in their training data. We show that we can produce comparable results on tweets with distant supervision.

#### 11

In order to train a classifier, supervised learning usually requires hand-labeled training data. With the large range of topics discussed on Twitter, it would be very difficult to manually collect enough data to train a sentiment classifier for tweets. Our solution is to use distant supervision, in which our training data consists of tweets with emoticons. This approach was introduced by Read [10]. The emoticons serve as noisy labels. For example, :) in a tweet indicates that the tweet contains positive sentiment and :( indicates that the tweet contains negative sentiment. With the help of the Twitter API, it is easy to extract large amounts of tweets with emoticons in them. This is a significant improvement over the many hours it may otherwise take to hand-label training data. We run classifiers trained on emoticon data against a test set of tweets (which may or may not have emoticons in them).

#### 12

We present the results of our experiments and our thoughts on how to further improve results. To help visualize the util-

#### 13

ity of a Twitter-based sentiment analysis tool, we also have a web application with our classifiers[1] . This can be used by individuals and companies that may want to research sentiment on any topic.

#### 14

For the purposes of our research, we define sentiment to be “a personal positive or negative feeling.” Table 1 shows some examples.

#### 15

Many times it is unclear if a tweet contains a sentiment. For these cases, we use the following litmus test: If the tweet could ever appear as a frontpage newspaper headline or as a sentence in Wikipedia, then it belongs in the neutral class. For example, the following tweet is considered neutral because it could have appeared as a newspaper headline, even though it projects an overall negative feeling about General Motors: _RT @Finance_ ~~_I_~~ _nfo Bankruptcy filing could put GM on road to profits (AP) http://cli.gs/9ua6Sb #Finance_ . In this research, we do not consider neutral tweets in our training or testing data. We only use positive or negative tweets. Many tweets do not have sentiment, so it is a current limitation of our research to not include the neutral class.

#### 16

## **1.2 Characteristics of Tweets**

#### 17

Twitter messages have many unique attributes, which differentiates our research from previous research:

#### 18

**Length** The maximum length of a Twitter message is 140 characters. From our training set, we calculate that the average length of a tweet is 14 words or 78 characters. This is very different from the previous sentiment classification research that focused on classifying longer bodies of work, such as movie reviews.

#### 19

**Data availability** Another difference is the magnitude of data available. With the Twitter API, it is very easy to collect millions of tweets for training. In past research, tests only consisted of thousands of training items.

#### 20

**Language model** Twitter users post messages from many different media, including their cell phones. The frequency of misspellings and slang in tweets is much higher than in other domains.
