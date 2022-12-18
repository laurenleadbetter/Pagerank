

# Pagerank Project

*Updates using pagerank2.py added under task 1 part 2, and task 2 part 2.


In this project, you will create a simple search engine for the website <https://www.lawfareblog.com>.
This website provides legal analysis on US national security issues.

**Due date:** Sunday, 18 September at midnight

**Computation:**
This project has low computational requirements.
You are not required to complete it on the lambda server (although you are welcome to if you'd like).

## Background

**Data:**

The `data` folder contains two files that store example "web graphs".
The file `small.csv.gz` contains the example graph from the *Deeper Inside Pagerank* paper.
This is a small graph, so we can manually inspect the contents of this file with the following command:
```
$ zcat data/small.csv.gz
source,target
1,2
1,3
3,1
3,2
3,5
4,5
4,6
5,6
5,4
6,4
```

> **Recall:**
> The `cat` terminal command outputs the contents of a file to stdout, and the `zcat` command first decompressed a gzipped file and then outputs the decompressed contents.

As you can see, the graph is stored as a CSV file.
The first line is a header,
and each subsequent line stores a single edge in the graph.
The first column contains the source node of the edge and the second column the target node.
The file is assumed to be sorted alphabetically.

The second data file `lawfareblog.csv.gz` contains the link structure for the lawfare blog.
Let's take a look at the first 10 of these lines:
```
$ zcat data/lawfareblog.csv.gz | head
source,target
www.lawfareblog.com/,www.lawfareblog.com/topic/interrogation
www.lawfareblog.com/,www.lawfareblog.com/upcoming-events
www.lawfareblog.com/,www.lawfareblog.com/
www.lawfareblog.com/,www.lawfareblog.com/our-comments-policy
www.lawfareblog.com/,www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
www.lawfareblog.com/,www.lawfareblog.com/topic/lawfare-research-paper-series
www.lawfareblog.com/,www.lawfareblog.com/topic/book-reviews
www.lawfareblog.com/,www.lawfareblog.com/documents-related-mueller-investigation
www.lawfareblog.com/,www.lawfareblog.com/topic/international-law-loac
```
You can see that in this file, the node names are URLs.
Semantically, each line corresponds to an HTML `<a>` tag that is contained in the source webpage and links to the target webpage.

We can use the following command to count the total number of links in the file:
```
$ zcat data/lawfareblog.csv.gz | wc -l
1610789
```
Since every link corresponds to a non-zero entry in the `P` matrix,
this is also the value of `nnz(P)`.
(Technically, we should subtract 1 from this value since the `wc -l` command also counts the header line, not just the data lines.)

To get the dimensions of `P`, we need to count the total number of nodes in the graph.
The following command achieves this by: decompressing the file, extracting the first column, removing all duplicate lines, then counting the results.
```
$ zcat data/lawfareblog.csv.gz | cut -f1 -d, | uniq | wc -l
25761
```
This matrix is large enough that computing matrix products for dense matrices takes several minutes on a single CPU.
Fortunately, however, the matrix is very sparse.
The following python code computes the fraction of entries in the matrix with non-zero values:
```
>>> 1610788 / (25760**2)
0.0024274297384360172
```
Thus, by using sparse matrix operations, we will be able to speed up the code significantly.

**Code:**

The `pagerank.py` file contains code for loading the graph CSV files and searching through their nodes for key phrases.
For example, you can perform a search for all nodes (i.e. urls) that mention the string `corona` with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --search_query=corona
```

> **NOTE:**
> It will take about 10 seconds to load and parse the data files.
> All the other computation happens essentially instantly.

Currently, the pagerank of the nodes is not currently being calculated correctly, and so the webpages are returned in an arbitrary order.
Your task in this assignment will be to fix these calculations in order to have the most important results (i.e. highest pagerank results) returned first.

## Task 1: the power method

Implement the `WebGraph.power_method` function in `pagerank.py` for computing the pagerank vector by fixing the `FIXME` annotation.

**Part 1:**

To check that your implementation is working,
you should run the program on the `data/small.csv.gz` graph.
For my implementation, I get the following output.
```
$ python3 pagerank.py --data=data/small.csv.gz --verbose
DEBUG:root:computing indices
DEBUG:root:computing values
DEBUG:root:i=0 residual=2.5629e-01
DEBUG:root:i=1 residual=1.1841e-01
DEBUG:root:i=2 residual=7.0701e-02
DEBUG:root:i=3 residual=3.1815e-02
DEBUG:root:i=4 residual=2.0497e-02
DEBUG:root:i=5 residual=1.0108e-02
DEBUG:root:i=6 residual=6.3716e-03
DEBUG:root:i=7 residual=3.4228e-03
DEBUG:root:i=8 residual=2.0879e-03
DEBUG:root:i=9 residual=1.1750e-03
DEBUG:root:i=10 residual=7.0131e-04
DEBUG:root:i=11 residual=4.0321e-04
DEBUG:root:i=12 residual=2.3800e-04
DEBUG:root:i=13 residual=1.3812e-04
DEBUG:root:i=14 residual=8.1083e-05
DEBUG:root:i=15 residual=4.7251e-05
DEBUG:root:i=16 residual=2.7704e-05
DEBUG:root:i=17 residual=1.6164e-05
DEBUG:root:i=18 residual=9.4778e-06
DEBUG:root:i=19 residual=5.5066e-06
DEBUG:root:i=20 residual=3.2042e-06
DEBUG:root:i=21 residual=1.8612e-06
DEBUG:root:i=22 residual=1.1283e-06
DEBUG:root:i=23 residual=6.1907e-07
INFO:root:rank=0 pagerank=6.6270e-01 url=4
INFO:root:rank=1 pagerank=5.2179e-01 url=6
INFO:root:rank=2 pagerank=4.1434e-01 url=5
INFO:root:rank=3 pagerank=2.3175e-01 url=2
INFO:root:rank=4 pagerank=1.8590e-01 url=3
INFO:root:rank=5 pagerank=1.6917e-01 url=1
```
Yours likely won't be identical (due to weird floating point issues), but it should be similar.
In particular, the ranking of the nodes/urls should be the same order.

> **NOTE:**
> The `--verbose` flag causes all of the lines beginning with `DEBUG` to be printed.
> By default, only lines beginning with `INFO` are printed.

**Part 2:**

The `pagerank.py` file has an option `--search_query`, which takes a string as a parameter.
If this argument is used, then the program returns all nodes that match the query string sorted according to their pagerank.
Essentially, this gives us the most important pages related to our query.

Again, you may not get the exact same results as me,
but you should get similar results to the examples I've shown below.
Verify that you do in fact get similar results.

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=8.9224e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=7.0390e-04 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=6.9153e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=6.7041e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=6.6256e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=6.5046e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=7 pagerank=6.3620e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=8 pagerank=6.1248e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
INFO:root:rank=9 pagerank=6.0187e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
INFO:root:rank=0 pagerank=5.7826e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=5.2338e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
INFO:root:rank=2 pagerank=5.1297e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
INFO:root:rank=3 pagerank=4.6599e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
INFO:root:rank=4 pagerank=4.5934e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=5 pagerank=4.3071e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=6 pagerank=4.0935e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
INFO:root:rank=7 pagerank=3.7591e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
INFO:root:rank=8 pagerank=3.4509e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
INFO:root:rank=9 pagerank=3.4484e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
INFO:root:rank=0 pagerank=4.5746e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
INFO:root:rank=1 pagerank=4.4174e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
INFO:root:rank=2 pagerank=2.6928e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=3 pagerank=1.9391e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
INFO:root:rank=4 pagerank=1.5452e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
INFO:root:rank=5 pagerank=1.5357e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
INFO:root:rank=6 pagerank=1.5258e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
INFO:root:rank=7 pagerank=1.4221e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=9 pagerank=1.1463e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force
```

**Part 3:**

The webgraph of lawfareblog.com (i.e. the `P` matrix) naturally contains a lot of structure.
For example, essentially all pages on the domain have links to the root page <https://lawfareblog.com/> and other "non-article" pages like <https://www.lawfareblog.com/topics> and <https://www.lawfareblog.com/subscribe-lawfare>.
These pages therefore have a large pagerank.
We can get a list of the pages with the largest pagerank by running

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz
INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/topics
INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
```

Most of these pages are not very interesting, however, because they are not articles,
and usually when we are performing a web search, we only want articles.

This raises the question: How can we find the most important articles filtering out the non-article pages?
The answer is to modify the `P` matrix by removing all links to non-article pages.

This raises another question: How do we know if a link is a non-article page?
Unfortunately, this is a hard question to answer with 100% accuracy,
but there are many methods that get us most of the way there.
One easy to implement method is to compute what's called the "in-link ratio" of each node (i.e. the total number of edges with the node as a target divided by the total number of nodes),
and then remove nodes from the search results with too-high of a ratio.
The intuition is that non-article pages often appear in the menu of a webpage, and so have links from almost all of the other webpages;
but article-webpages are unlikely to appear on a menu and so will only have a small number of links from other webpages.
The `--filter_ratio` parameter causes the code to remove all pages that have an in-link ratio larger than the provided value.

Using this option, we can estimate the most important articles on the domain with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
```
Notice that the urls in this list look much more like articles than the urls in the previous list.

When Google calculates their `P` matrix for the web,
they use a similar (but much more complicated) process to modify the `P` matrix in order to reduce spam results.
The exact formula they use is a jealously guarded secret that they update continuously.

In the case above, notice that we have accidentally removed the blog's most popular article (<www.lawfareblog.com/snowden-revelations>).
The blog editors believed that Snowden's revelations about NSA spying are so important that they directly put a link to the article on the menu.
So every single webpage in the domain links to the Snowden article,
and our "anti-spam" `--filter-ratio` argument removed this article from the list.
In general, it is a challenging open problem to remove spam from pagerank results,
and all current solutions rely on careful human tuning and still have lots of false positives and false negatives.

**Part 4:**

Recall from the reading that the runtime of pagerank depends heavily on the eigengap of the `\bar\bar P` matrix,
and that this eigengap is bounded by the alpha parameter.

Run the following four commands:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
```
You should notice that the last command takes considerably more iterations to compute the pagerank vector.
(My code takes 685 iterations for this call, and about 10 iterations for all the others.)

This raises the question: Why does the second command (with the `--alpha` option but without the `--filter_ratio`) option not take a long time to run?
The answer is that the `P` graph for <https://www.lawfareblog.com> naturally has a large eigengap and so is fast to compute for all alpha values,
but the modified graph does not have a large eigengap and so requires a small alpha for fast convergence.

Changing the value of alpha also gives us very different pagerank rankings.
For example, 
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.0552e-01 url=www.lawfareblog.com/cost-using-zero-days
INFO:root:rank=3 pagerank=3.1755e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
INFO:root:rank=4 pagerank=2.2040e-02 url=www.lawfareblog.com/events
INFO:root:rank=5 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
INFO:root:rank=6 pagerank=1.6026e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
INFO:root:rank=7 pagerank=1.6023e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
INFO:root:rank=8 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
INFO:root:rank=9 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
```

Which of these rankings is better is entirely subjective,
and the only way to know if you have the "best" alpha for your application is to try several variations and see what is best.
If large alphas are good for your application, you can see that there is a trade-off between quality answers and algorithmic runtime.
We'll be exploring this trade-off more formally in class over the rest of the semester.

## Task 2: the personalization vector

The most interesting applications of pagerank involve the personalization vector.
Implement the `WebGraph.make_personalization_vector` function so that it outputs a personalization vector tuned for the input query.
The pseudocode for the function is:
```
for each index in the personalization vector:
    get the url for the index (see the _index_to_url function)
    check if the url satisfies the input query (see the url_satisfies_query function)
    if so, set the corresponding index to one
normalize the vector
```

**Part 1:**

The command line argument `--personalization_vector_query` will use the function you created above to augment your search with a custom personalization vector.
If you've implemented the function correctly,
you should get results similar to:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
```

Notice that these results are significantly different than when using the `--search_query` option:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --search_query='corona'
INFO:root:rank=0 pagerank=8.1320e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=1 pagerank=7.7908e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=2 pagerank=5.2262e-03 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
INFO:root:rank=3 pagerank=3.9584e-03 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=4 pagerank=3.8114e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=5 pagerank=3.3973e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=6 pagerank=3.3633e-03 url=www.lawfareblog.com/cyberlaw-podcast-how-israel-fighting-coronavirus
INFO:root:rank=7 pagerank=3.3557e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=8 pagerank=3.2160e-03 url=www.lawfareblog.com/congress-needs-coronavirus-failsafe-its-too-late
INFO:root:rank=9 pagerank=3.1036e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
```

Which results are better?
Again, that depends on what you mean by "better."
With the `--personalization_vector_query` option,
a webpage is important only if other coronavirus webpages also think it's important;
with the `--search_query` option,
a webpage is important if any other webpage thinks it's important.
You'll notice that in the later example, many of the webpages are about Congressional proceedings related to the coronavirus.
From a strictly coronavirus perspective, these are not very important webpages.
But in the broader context of national security, these are very important webpages.

Google engineers spend TONs of time fine-tuning their pagerank personalization vectors to remove spam webpages.
Exactly how they do this is another one of their secrets that they don't publicly talk about.

**Part 2:**

Another use of the `--personalization_vector_query` option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus.
This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their `corona` importance,
but removes webpages mentioning `corona` from the results.
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america
```
You can see that there are many urls about concepts that are obviously related like "covid", "clinical trials", and "quarantine",
but this algorithm also finds articles about Chinese propaganda and Trump's policy decisions.
Both of these articles are highly relevant to coronavirus discussions,
but a simple keyword search for corona or related terms would not find these articles.

<!--
**Part 3:**

Select another topic related to national security.
You should experiment with a national security topic other than the coronavirus.
For example, find out what articles are important to the `iran` topic but do not contain the word `iran`.
Your goal should be to discover what topics that www.lawfareblog.com considers to be related to the national security topic you choose.
-->

## Submission

1. Create a new repo on github (not a fork of this repo).

1. Run the following commands, and paste their output into the code blocks below.
   
   Task 1, part 1:
   ```
   $ python3 pagerank.py --data=data/small.csv.gz --verbose 
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=0.3775096535682678
    DEBUG:root:i=1 residual=0.3134882152080536
    DEBUG:root:i=2 residual=0.2756592035293579
    DEBUG:root:i=3 residual=0.21698090434074402
    DEBUG:root:i=4 residual=0.18984203040599823
    DEBUG:root:i=5 residual=0.15531450510025024
    DEBUG:root:i=6 residual=0.1326625943183899
    DEBUG:root:i=7 residual=0.11062272638082504
    DEBUG:root:i=8 residual=0.09351350367069244
    DEBUG:root:i=9 residual=0.07847187668085098
    DEBUG:root:i=10 residual=0.06611086428165436
    DEBUG:root:i=11 residual=0.05558090656995773
    DEBUG:root:i=12 residual=0.046779245138168335
    DEBUG:root:i=13 residual=0.03934890031814575
    DEBUG:root:i=14 residual=0.033108800649642944
    DEBUG:root:i=15 residual=0.027853814885020256
    DEBUG:root:i=16 residual=0.023434916511178017
    DEBUG:root:i=17 residual=0.019716285169124603
    DEBUG:root:i=18 residual=0.01658768206834793
    DEBUG:root:i=19 residual=0.01395586971193552
    DEBUG:root:i=20 residual=0.011741566471755505
    DEBUG:root:i=21 residual=0.00987835694104433
    DEBUG:root:i=22 residual=0.00831097736954689
    DEBUG:root:i=23 residual=0.006992337293922901
    DEBUG:root:i=24 residual=0.00588282709941268
    DEBUG:root:i=25 residual=0.00494928564876318
    DEBUG:root:i=26 residual=0.0041639236733317375
    DEBUG:root:i=27 residual=0.003503408282995224
    DEBUG:root:i=28 residual=0.0029473905451595783
    DEBUG:root:i=29 residual=0.002479690359905362
    DEBUG:root:i=30 residual=0.0020863786339759827
    DEBUG:root:i=31 residual=0.001755346660502255
    DEBUG:root:i=32 residual=0.0014766508247703314
    DEBUG:root:i=33 residual=0.0012425254099071026
    DEBUG:root:i=34 residual=0.0010451258858665824
    DEBUG:root:i=35 residual=0.0008794772438704967
    DEBUG:root:i=36 residual=0.0007398196030408144
    DEBUG:root:i=37 residual=0.0006225692923180759
    DEBUG:root:i=38 residual=0.0005236482247710228
    DEBUG:root:i=39 residual=0.0004406241059768945
    DEBUG:root:i=40 residual=0.00037083958159200847
    DEBUG:root:i=41 residual=0.0003118595341220498
    DEBUG:root:i=42 residual=0.0002623549662530422
    DEBUG:root:i=43 residual=0.00022077768517192453
    DEBUG:root:i=44 residual=0.00018579674360807985
    DEBUG:root:i=45 residual=0.00015626229287590832
    DEBUG:root:i=46 residual=0.00013155356282368302
    DEBUG:root:i=47 residual=0.00011065449507441372
    DEBUG:root:i=48 residual=9.283208783017471e-05
    DEBUG:root:i=49 residual=7.830880349501967e-05
    DEBUG:root:i=50 residual=6.584281072719023e-05
    DEBUG:root:i=51 residual=5.537284960155375e-05
    DEBUG:root:i=52 residual=4.678148252423853e-05
    DEBUG:root:i=53 residual=3.9097874832805246e-05
    DEBUG:root:i=54 residual=3.2944819395197555e-05
    DEBUG:root:i=55 residual=2.7851461709360592e-05
    DEBUG:root:i=56 residual=2.3248065190273337e-05
    DEBUG:root:i=57 residual=1.966203490155749e-05
    DEBUG:root:i=58 residual=1.6472085917484947e-05
    DEBUG:root:i=59 residual=1.4015268789080437e-05
    DEBUG:root:i=60 residual=1.1800665561167989e-05
    DEBUG:root:i=61 residual=9.697989298729226e-06
    DEBUG:root:i=62 residual=8.280595466203522e-06
    DEBUG:root:i=63 residual=7.129734513000585e-06
    DEBUG:root:i=64 residual=5.845966825290816e-06
    DEBUG:root:i=65 residual=4.963138053426519e-06
    DEBUG:root:i=66 residual=4.2077249418071005e-06
    DEBUG:root:i=67 residual=3.4983154364454094e-06
    DEBUG:root:i=68 residual=2.992129338963423e-06
    DEBUG:root:i=69 residual=2.5033950805664062e-06
    DEBUG:root:i=70 residual=2.214214191553765e-06
    DEBUG:root:i=71 residual=1.955177822310361e-06
    DEBUG:root:i=72 residual=1.3902072169003077e-06
    DEBUG:root:i=73 residual=1.244581540049694e-06
    DEBUG:root:i=74 residual=9.97376446321141e-07
    INFO:root:rank=0 pagerank=2.1634e+00 url=4
    INFO:root:rank=1 pagerank=1.6664e+00 url=6
    INFO:root:rank=2 pagerank=1.2402e+00 url=5
    INFO:root:rank=3 pagerank=4.5712e-01 url=2
    INFO:root:rank=4 pagerank=3.5620e-01 url=3
    INFO:root:rank=5 pagerank=3.2078e-01 url=1
   ```

   Task 1, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
    INFO:root:rank=0 pagerank=4.5865e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
    INFO:root:rank=1 pagerank=4.0464e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
    INFO:root:rank=2 pagerank=2.6118e-03 url=www.lawfareblog.com/britains-coronavirus-response
    INFO:root:rank=3 pagerank=2.5392e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
    INFO:root:rank=4 pagerank=2.3560e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
    INFO:root:rank=5 pagerank=2.2897e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
    INFO:root:rank=6 pagerank=2.2729e-03 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
    INFO:root:rank=7 pagerank=2.2522e-03 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
    INFO:root:rank=8 pagerank=2.1880e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
    INFO:root:rank=9 pagerank=2.0341e-03 url=www.lawfareblog.com/cyberlaw-podcast-how-israel-fighting-coronavirus

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
   INFO:root:rank=0 pagerank=6.6250e-02 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
    INFO:root:rank=1 pagerank=6.0200e-02 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=2 pagerank=3.4972e-02 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
    INFO:root:rank=3 pagerank=3.2196e-02 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
    INFO:root:rank=4 pagerank=3.0974e-02 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
    INFO:root:rank=5 pagerank=2.8463e-02 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
    INFO:root:rank=6 pagerank=2.5255e-02 url=www.lawfareblog.com/why-trump-cant-buy-greenland
    INFO:root:rank=7 pagerank=2.2459e-02 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
    INFO:root:rank=8 pagerank=2.1464e-02 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
    INFO:root:rank=9 pagerank=2.1105e-02 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
    INFO:root:rank=0 pagerank=6.6138e-02 url=www.lawfareblog.com/praise-presidents-iran-tweets
    INFO:root:rank=1 pagerank=2.9202e-02 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
    INFO:root:rank=2 pagerank=1.7711e-02 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
    INFO:root:rank=3 pagerank=1.4606e-02 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
    INFO:root:rank=4 pagerank=8.4519e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
    INFO:root:rank=5 pagerank=8.3997e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
    INFO:root:rank=6 pagerank=8.2589e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
    INFO:root:rank=7 pagerank=8.0568e-03 url=www.lawfareblog.com/trump-moves-cut-irans-oil-revenues-whats-his-endgame
    INFO:root:rank=8 pagerank=7.1946e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
    INFO:root:rank=9 pagerank=5.9410e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
   ```


**Update using Pagerank2.py**

    python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
    INFO:gensim.models.keyedvectors:precomputing L2-norms of word weight vectors
    INFO:root:rank=0 pagerank=5.7826e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=5.2338e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
    INFO:root:rank=2 pagerank=5.1297e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
    INFO:root:rank=3 pagerank=4.6599e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
    INFO:root:rank=4 pagerank=4.5934e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
    INFO:root:rank=5 pagerank=4.3071e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
    INFO:root:rank=6 pagerank=4.0935e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
    INFO:root:rank=7 pagerank=3.7591e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
    INFO:root:rank=8 pagerank=3.4509e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
    INFO:root:rank=9 pagerank=3.4484e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors



   Task 1, part 3:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz
   INFO:root:rank=0 pagerank=8.4165e+00 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=1 pagerank=8.4165e+00 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=8.4165e+00 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=3 pagerank=8.4165e+00 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=8.4165e+00 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=8.4165e+00 url=www.lawfareblog.com/masthead
    INFO:root:rank=6 pagerank=8.4165e+00 url=www.lawfareblog.com/topics
    INFO:root:rank=7 pagerank=8.4165e+00 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=8.4165e+00 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=8.4165e+00 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
   INFO:root:rank=0 pagerank=4.6096e+00 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=2.9870e+00 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 pagerank=2.9672e+00 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 pagerank=2.0175e+00 url=www.lawfareblog.com/senate-examines-threats-homeland
    INFO:root:rank=4 pagerank=1.8771e+00 url=www.lawfareblog.com/what-make-first-day-impeachment-hearings
    INFO:root:rank=5 pagerank=1.8764e+00 url=www.lawfareblog.com/livestream-house-armed-services-committee-hearing-f-35-program
    INFO:root:rank=6 pagerank=1.8695e+00 url=www.lawfareblog.com/whats-house-resolution-impeachment
    INFO:root:rank=7 pagerank=1.7657e+00 url=www.lawfareblog.com/congress-us-policy-toward-syria-and-turkey-overview-recent-hearings
    INFO:root:rank=8 pagerank=1.6809e+00 url=www.lawfareblog.com/summary-david-holmess-deposition-testimony
    INFO:root:rank=9 pagerank=9.8355e-01 url=www.lawfareblog.com/events
   ```

   Task 1, part 4:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
   DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=20.521549224853516
    DEBUG:root:i=1 residual=6.110842227935791
    DEBUG:root:i=2 residual=1.9216899871826172
    DEBUG:root:i=3 residual=0.5883616805076599
    DEBUG:root:i=4 residual=0.17544545233249664
    DEBUG:root:i=5 residual=0.05154035612940788
    DEBUG:root:i=6 residual=0.014946999028325081
    DEBUG:root:i=7 residual=0.004281123634427786
    DEBUG:root:i=8 residual=0.0011693136766552925
    DEBUG:root:i=9 residual=0.00026119541143998504
    DEBUG:root:i=10 residual=3.841046054731123e-05
    DEBUG:root:i=11 residual=6.379340629791841e-05
    DEBUG:root:i=12 residual=7.276752876350656e-05
    DEBUG:root:i=13 residual=6.609541014768183e-05
    DEBUG:root:i=14 residual=5.948194666416384e-05
    DEBUG:root:i=15 residual=5.2871771913487464e-05
    DEBUG:root:i=16 residual=4.2959396523656324e-05
    DEBUG:root:i=17 residual=3.634978202171624e-05
    DEBUG:root:i=18 residual=2.9742965125478804e-05
    DEBUG:root:i=19 residual=2.9738816010649316e-05
    DEBUG:root:i=20 residual=2.3133206923375838e-05
    DEBUG:root:i=21 residual=1.9827675714623183e-05
    DEBUG:root:i=22 residual=1.6522120859008282e-05
    DEBUG:root:i=23 residual=1.3217999367043376e-05
    DEBUG:root:i=24 residual=9.915413102135062e-06
    DEBUG:root:i=25 residual=1.3216712432040367e-05
    DEBUG:root:i=26 residual=6.611635853914777e-06
    DEBUG:root:i=27 residual=6.608514922845643e-06
    DEBUG:root:i=28 residual=9.911609595292248e-06
    DEBUG:root:i=29 residual=3.3084943424910307e-06
    DEBUG:root:i=30 residual=6.608346211578464e-06
    DEBUG:root:i=31 residual=6.608358489756938e-06
    DEBUG:root:i=32 residual=3.3058420285669854e-06
    DEBUG:root:i=33 residual=3.304204938103794e-06
    DEBUG:root:i=34 residual=1.2050155362430814e-07
    INFO:root:rank=0 pagerank=8.4165e+00 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=1 pagerank=8.4165e+00 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=8.4165e+00 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=3 pagerank=8.4165e+00 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=8.4165e+00 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=8.4165e+00 url=www.lawfareblog.com/masthead
    INFO:root:rank=6 pagerank=8.4165e+00 url=www.lawfareblog.com/topics
    INFO:root:rank=7 pagerank=8.4165e+00 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=8.4165e+00 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=8.4165e+00 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site


   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
   DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=24.140165328979492
    DEBUG:root:i=1 residual=8.458998680114746
    DEBUG:root:i=2 residual=3.128697395324707
    DEBUG:root:i=3 residual=1.1249889135360718
    DEBUG:root:i=4 residual=0.3952803313732147
    DEBUG:root:i=5 residual=0.13689270615577698
    DEBUG:root:i=6 residual=0.04697414115071297
    DEBUG:root:i=7 residual=0.015963083133101463
    DEBUG:root:i=8 residual=0.005272228736430407
    DEBUG:root:i=9 residual=0.001591996056959033
    DEBUG:root:i=10 residual=0.0003603595250751823
    DEBUG:root:i=11 residual=0.00015778069791849703
    DEBUG:root:i=12 residual=0.00026254155091010034
    DEBUG:root:i=13 residual=0.00029754918068647385
    DEBUG:root:i=14 residual=0.0003007303166668862
    DEBUG:root:i=15 residual=0.00031062401831150055
    DEBUG:root:i=16 residual=0.000323840300552547
    DEBUG:root:i=17 residual=0.00032054336043074727
    DEBUG:root:i=18 residual=0.0003172410069964826
    DEBUG:root:i=19 residual=0.0003172381257172674
    DEBUG:root:i=20 residual=0.0003172381257172674
    DEBUG:root:i=21 residual=0.0003172381257172674
    DEBUG:root:i=22 residual=0.00031723809661343694
    DEBUG:root:i=23 residual=0.0003172381257172674
    DEBUG:root:i=24 residual=0.000317238038405776
    DEBUG:root:i=25 residual=0.0003139368782285601
    DEBUG:root:i=26 residual=0.00031723672873340547
    DEBUG:root:i=27 residual=0.0003139339678455144
    DEBUG:root:i=28 residual=0.00031723672873340547
    DEBUG:root:i=29 residual=0.0003172381839249283
    DEBUG:root:i=30 residual=0.00031723809661343694
    DEBUG:root:i=31 residual=0.0003172380675096065
    DEBUG:root:i=32 residual=0.0003172381839249283
    DEBUG:root:i=33 residual=0.0003172380675096065
    DEBUG:root:i=34 residual=0.0003172395227011293
    DEBUG:root:i=35 residual=0.000317238038405776
    DEBUG:root:i=36 residual=0.0003172381839249283
    DEBUG:root:i=37 residual=0.0003172381257172674
    DEBUG:root:i=38 residual=0.0003172381257172674
    DEBUG:root:i=39 residual=0.0003172381839249283
    DEBUG:root:i=40 residual=0.000317238038405776
    DEBUG:root:i=41 residual=0.0003172394644934684
    DEBUG:root:i=42 residual=0.0003172381257172674
    DEBUG:root:i=43 residual=0.0003139341133646667
    DEBUG:root:i=44 residual=0.0003172366414219141
    DEBUG:root:i=45 residual=0.0003172381839249283
    DEBUG:root:i=46 residual=0.00031723809661343694
    DEBUG:root:i=47 residual=0.0003172380675096065
    DEBUG:root:i=48 residual=0.0003205407992936671
    DEBUG:root:i=49 residual=0.0003205436223652214
    DEBUG:root:i=50 residual=0.0003172396682202816
    DEBUG:root:i=51 residual=0.0003172381257172674
    DEBUG:root:i=52 residual=0.0003172395227011293
    DEBUG:root:i=53 residual=0.0003172381257172674
    DEBUG:root:i=54 residual=0.00031723809661343694
    DEBUG:root:i=55 residual=0.0003172381257172674
    DEBUG:root:i=56 residual=0.00031723809661343694
    DEBUG:root:i=57 residual=0.0003172381257172674
    DEBUG:root:i=58 residual=0.000317238038405776
    DEBUG:root:i=59 residual=0.0003172395518049598
    DEBUG:root:i=60 residual=0.0003172381257172674
    DEBUG:root:i=61 residual=0.00031723815482109785
    DEBUG:root:i=62 residual=0.0003172381257172674
    DEBUG:root:i=63 residual=0.0003172380675096065
    DEBUG:root:i=64 residual=0.00031723667052574456
    DEBUG:root:i=65 residual=0.00031723949359729886
    DEBUG:root:i=66 residual=0.0003172381257172674
    DEBUG:root:i=67 residual=0.0003172381839249283
    DEBUG:root:i=68 residual=0.0003172382421325892
    DEBUG:root:i=69 residual=0.000317238038405776
    DEBUG:root:i=70 residual=0.0003172381257172674
    DEBUG:root:i=71 residual=0.00031723809661343694
    DEBUG:root:i=72 residual=0.0003139368782285601
    DEBUG:root:i=73 residual=0.0003139341133646667
    DEBUG:root:i=74 residual=0.0003172352153342217
    DEBUG:root:i=75 residual=0.0003172394644934684
    DEBUG:root:i=76 residual=0.0003172381257172674
    DEBUG:root:i=77 residual=0.0003172381839249283
    DEBUG:root:i=78 residual=0.0003172380675096065
    DEBUG:root:i=79 residual=0.000320540857501328
    DEBUG:root:i=80 residual=0.0003172408032696694
    DEBUG:root:i=81 residual=0.00031723821302875876
    DEBUG:root:i=82 residual=0.00031723809661343694
    DEBUG:root:i=83 residual=0.00031723672873340547
    DEBUG:root:i=84 residual=0.0003172380675096065
    DEBUG:root:i=85 residual=0.0003172381839249283
    DEBUG:root:i=86 residual=0.00031723949359729886
    DEBUG:root:i=87 residual=0.00031723809661343694
    DEBUG:root:i=88 residual=0.0003172382421325892
    DEBUG:root:i=89 residual=0.0003172380675096065
    DEBUG:root:i=90 residual=0.0003139354521408677
    DEBUG:root:i=91 residual=0.0003172366414219141
    DEBUG:root:i=92 residual=0.0003172381257172674
    DEBUG:root:i=93 residual=0.0003172381257172674
    DEBUG:root:i=94 residual=0.0003172395518049598
    DEBUG:root:i=95 residual=0.00031723821302875876
    DEBUG:root:i=96 residual=0.0003172380675096065
    DEBUG:root:i=97 residual=0.000317238038405776
    DEBUG:root:i=98 residual=0.00031723809661343694
    DEBUG:root:i=99 residual=0.0003172381257172674
    DEBUG:root:i=100 residual=0.000317238038405776
    DEBUG:root:i=101 residual=0.0003172381839249283
    DEBUG:root:i=102 residual=0.000317238038405776
    DEBUG:root:i=103 residual=0.0003172381257172674
    DEBUG:root:i=104 residual=0.0003205407992936671
    DEBUG:root:i=105 residual=0.0003172395518049598
    DEBUG:root:i=106 residual=0.000320542196277529
    DEBUG:root:i=107 residual=0.00031723958090879023
    DEBUG:root:i=108 residual=0.0003172381839249283
    DEBUG:root:i=109 residual=0.0003172381257172674
    DEBUG:root:i=110 residual=0.0003172381257172674
    DEBUG:root:i=111 residual=0.000320542196277529
    DEBUG:root:i=112 residual=0.0003172395518049598
    DEBUG:root:i=113 residual=0.00031723809661343694
    DEBUG:root:i=114 residual=0.0003172380675096065
    DEBUG:root:i=115 residual=0.0003172381839249283
    DEBUG:root:i=116 residual=0.0003172395518049598
    DEBUG:root:i=117 residual=0.00031723815482109785
    DEBUG:root:i=118 residual=0.00031723815482109785
    DEBUG:root:i=119 residual=0.0003172380675096065
    DEBUG:root:i=120 residual=0.0003172381257172674
    DEBUG:root:i=121 residual=0.0003172380675096065
    DEBUG:root:i=122 residual=0.0003172395518049598
    DEBUG:root:i=123 residual=0.00031723809661343694
    DEBUG:root:i=124 residual=0.00031723815482109785
    DEBUG:root:i=125 residual=0.00031723809661343694
    DEBUG:root:i=126 residual=0.00031723675783723593
    DEBUG:root:i=127 residual=0.00031723809661343694
    DEBUG:root:i=128 residual=0.0003172381257172674
    DEBUG:root:i=129 residual=0.0003139368782285601
    DEBUG:root:i=130 residual=0.0003172366414219141
    DEBUG:root:i=131 residual=0.0003139354521408677
    DEBUG:root:i=132 residual=0.00031723672873340547
    DEBUG:root:i=133 residual=0.0003172380675096065
    DEBUG:root:i=134 residual=0.00031723809661343694
    DEBUG:root:i=135 residual=0.00031723821302875876
    DEBUG:root:i=136 residual=0.00031723800930194557
    DEBUG:root:i=137 residual=0.0003172381257172674
    DEBUG:root:i=138 residual=0.0003172394644934684
    DEBUG:root:i=139 residual=0.0003172381839249283
    DEBUG:root:i=140 residual=0.00031723821302875876
    DEBUG:root:i=141 residual=0.000317238038405776
    DEBUG:root:i=142 residual=0.0003172381839249283
    DEBUG:root:i=143 residual=0.0003172366414219141
    DEBUG:root:i=144 residual=0.0003172381257172674
    DEBUG:root:i=145 residual=0.0003172395518049598
    DEBUG:root:i=146 residual=0.00031723815482109785
    DEBUG:root:i=147 residual=0.00031723809661343694
    DEBUG:root:i=148 residual=0.00031723809661343694
    DEBUG:root:i=149 residual=0.0003172381257172674
    DEBUG:root:i=150 residual=0.000317238038405776
    DEBUG:root:i=151 residual=0.00032384495716542006
    DEBUG:root:i=152 residual=0.00032054356415756047
    DEBUG:root:i=153 residual=0.00031724097789265215
    DEBUG:root:i=154 residual=0.0003172381257172674
    DEBUG:root:i=155 residual=0.0003139355103485286
    DEBUG:root:i=156 residual=0.0003172367869410664
    DEBUG:root:i=157 residual=0.00031723800930194557
    DEBUG:root:i=158 residual=0.0003172381257172674
    DEBUG:root:i=159 residual=0.00031723809661343694
    DEBUG:root:i=160 residual=0.0003172381257172674
    DEBUG:root:i=161 residual=0.0003172394644934684
    DEBUG:root:i=162 residual=0.0003172381839249283
    DEBUG:root:i=163 residual=0.0003139355103485286
    DEBUG:root:i=164 residual=0.00031723675783723593
    DEBUG:root:i=165 residual=0.0003205394314136356
    DEBUG:root:i=166 residual=0.00031723949359729886
    DEBUG:root:i=167 residual=0.0003172381257172674
    DEBUG:root:i=168 residual=0.00031723949359729886
    DEBUG:root:i=169 residual=0.0003172381257172674
    DEBUG:root:i=170 residual=0.0003172380675096065
    DEBUG:root:i=171 residual=0.0003172381839249283
    DEBUG:root:i=172 residual=0.0003172381257172674
    DEBUG:root:i=173 residual=0.00031723809661343694
    DEBUG:root:i=174 residual=0.00031723949359729886
    DEBUG:root:i=175 residual=0.0003205407992936671
    DEBUG:root:i=176 residual=0.0003172395518049598
    DEBUG:root:i=177 residual=0.00031393542303703725
    DEBUG:root:i=178 residual=0.00031723675783723593
    DEBUG:root:i=179 residual=0.0003172381257172674
    DEBUG:root:i=180 residual=0.00031723809661343694
    DEBUG:root:i=181 residual=0.0003172394644934684
    DEBUG:root:i=182 residual=0.00032054088660515845
    DEBUG:root:i=183 residual=0.0003172395518049598
    DEBUG:root:i=184 residual=0.0003172381839249283
    DEBUG:root:i=185 residual=0.0003172381257172674
    DEBUG:root:i=186 residual=0.0003172380675096065
    DEBUG:root:i=187 residual=0.0003172381257172674
    DEBUG:root:i=188 residual=0.00031393542303703725
    DEBUG:root:i=189 residual=0.00031723687425255775
    DEBUG:root:i=190 residual=0.00031723809661343694
    DEBUG:root:i=191 residual=0.00031723809661343694
    DEBUG:root:i=192 residual=0.0003172380675096065
    DEBUG:root:i=193 residual=0.00031393536482937634
    DEBUG:root:i=194 residual=0.00031723672873340547
    DEBUG:root:i=195 residual=0.0003172395227011293
    DEBUG:root:i=196 residual=0.0003172381839249283
    DEBUG:root:i=197 residual=0.000317238038405776
    DEBUG:root:i=198 residual=0.0003172381257172674
    DEBUG:root:i=199 residual=0.00031723809661343694
    DEBUG:root:i=200 residual=0.00031723815482109785
    DEBUG:root:i=201 residual=0.0003172380675096065
    DEBUG:root:i=202 residual=0.0003172395227011293
    DEBUG:root:i=203 residual=0.00031723815482109785
    DEBUG:root:i=204 residual=0.0003172381257172674
    DEBUG:root:i=205 residual=0.0003172367869410664
    DEBUG:root:i=206 residual=0.0003172380675096065
    DEBUG:root:i=207 residual=0.0003172381257172674
    DEBUG:root:i=208 residual=0.00031723943538963795
    DEBUG:root:i=209 residual=0.0003172381257172674
    DEBUG:root:i=210 residual=0.00031723809661343694
    DEBUG:root:i=211 residual=0.00032054082839749753
    DEBUG:root:i=212 residual=0.00031723949359729886
    DEBUG:root:i=213 residual=0.0003172396682202816
    DEBUG:root:i=214 residual=0.0003172381257172674
    DEBUG:root:i=215 residual=0.0003172380675096065
    DEBUG:root:i=216 residual=0.0003172381257172674
    DEBUG:root:i=217 residual=0.00031723809661343694
    DEBUG:root:i=218 residual=0.0003172381839249283
    DEBUG:root:i=219 residual=0.00031393542303703725
    DEBUG:root:i=220 residual=0.0003172366414219141
    DEBUG:root:i=221 residual=0.0003172394644934684
    DEBUG:root:i=222 residual=0.0003172381839249283
    DEBUG:root:i=223 residual=0.00031393548124469817
    DEBUG:root:i=224 residual=0.0003172353026457131
    DEBUG:root:i=225 residual=0.000317238038405776
    DEBUG:root:i=226 residual=0.000317238038405776
    DEBUG:root:i=227 residual=0.0003172381839249283
    DEBUG:root:i=228 residual=0.0003172381257172674
    DEBUG:root:i=229 residual=0.0003172395227011293
    DEBUG:root:i=230 residual=0.0003139354521408677
    DEBUG:root:i=231 residual=0.00031723667052574456
    DEBUG:root:i=232 residual=0.000317238038405776
    DEBUG:root:i=233 residual=0.0003172381257172674
    DEBUG:root:i=234 residual=0.0003172381257172674
    DEBUG:root:i=235 residual=0.0003172381257172674
    DEBUG:root:i=236 residual=0.000317238038405776
    DEBUG:root:i=237 residual=0.000320542196277529
    DEBUG:root:i=238 residual=0.00031723963911645114
    DEBUG:root:i=239 residual=0.0003172381257172674
    DEBUG:root:i=240 residual=0.0003172381839249283
    DEBUG:root:i=241 residual=0.0003172381257172674
    DEBUG:root:i=242 residual=0.0003172380675096065
    DEBUG:root:i=243 residual=0.00031723815482109785
    DEBUG:root:i=244 residual=0.0003172381257172674
    DEBUG:root:i=245 residual=0.00031723821302875876
    DEBUG:root:i=246 residual=0.000317238038405776
    DEBUG:root:i=247 residual=0.0003172380675096065
    DEBUG:root:i=248 residual=0.00031723809661343694
    DEBUG:root:i=249 residual=0.0003172395227011293
    DEBUG:root:i=250 residual=0.0003172381257172674
    DEBUG:root:i=251 residual=0.0003172381257172674
    DEBUG:root:i=252 residual=0.0003172381257172674
    DEBUG:root:i=253 residual=0.000317238038405776
    DEBUG:root:i=254 residual=0.0003172381257172674
    DEBUG:root:i=255 residual=0.0003172395227011293
    DEBUG:root:i=256 residual=0.0003172381839249283
    DEBUG:root:i=257 residual=0.00031723809661343694
    DEBUG:root:i=258 residual=0.0003172380675096065
    DEBUG:root:i=259 residual=0.000317238038405776
    DEBUG:root:i=260 residual=0.0003139355103485286
    DEBUG:root:i=261 residual=0.00031723672873340547
    DEBUG:root:i=262 residual=0.00031723675783723593
    DEBUG:root:i=263 residual=0.00031723958090879023
    DEBUG:root:i=264 residual=0.00031723809661343694
    DEBUG:root:i=265 residual=0.0003172381257172674
    DEBUG:root:i=266 residual=0.0003172380675096065
    DEBUG:root:i=267 residual=0.0003172381839249283
    DEBUG:root:i=268 residual=0.00031723809661343694
    DEBUG:root:i=269 residual=0.0003172381257172674
    DEBUG:root:i=270 residual=0.0003205421380698681
    DEBUG:root:i=271 residual=0.0003172395518049598
    DEBUG:root:i=272 residual=0.0003172381257172674
    DEBUG:root:i=273 residual=0.0003172381257172674
    DEBUG:root:i=274 residual=0.0003172381839249283
    DEBUG:root:i=275 residual=0.00031723949359729886
    DEBUG:root:i=276 residual=0.00031723815482109785
    DEBUG:root:i=277 residual=0.00031723809661343694
    DEBUG:root:i=278 residual=0.0003172381257172674
    DEBUG:root:i=279 residual=0.00031393542303703725
    DEBUG:root:i=280 residual=0.00031723672873340547
    DEBUG:root:i=281 residual=0.0003172381257172674
    DEBUG:root:i=282 residual=0.00031723667052574456
    DEBUG:root:i=283 residual=0.00031723958090879023
    DEBUG:root:i=284 residual=0.0003172381257172674
    DEBUG:root:i=285 residual=0.0003172381257172674
    DEBUG:root:i=286 residual=0.000317238038405776
    DEBUG:root:i=287 residual=0.0003172381839249283
    DEBUG:root:i=288 residual=0.0003172381257172674
    DEBUG:root:i=289 residual=0.0003172394644934684
    DEBUG:root:i=290 residual=0.00031723815482109785
    DEBUG:root:i=291 residual=0.0003172381257172674
    DEBUG:root:i=292 residual=0.00031393536482937634
    DEBUG:root:i=293 residual=0.000317236699629575
    DEBUG:root:i=294 residual=0.0003172381257172674
    DEBUG:root:i=295 residual=0.000317238038405776
    DEBUG:root:i=296 residual=0.0003172381839249283
    DEBUG:root:i=297 residual=0.0003172394062858075
    DEBUG:root:i=298 residual=0.0003139355103485286
    DEBUG:root:i=299 residual=0.0003139339678455144
    DEBUG:root:i=300 residual=0.000317235360853374
    DEBUG:root:i=301 residual=0.0003205407701898366
    DEBUG:root:i=302 residual=0.0003172395227011293
    DEBUG:root:i=303 residual=0.0003172381257172674
    DEBUG:root:i=304 residual=0.00031723809661343694
    DEBUG:root:i=305 residual=0.0003172395518049598
    DEBUG:root:i=306 residual=0.0003172381257172674
    DEBUG:root:i=307 residual=0.000320540857501328
    DEBUG:root:i=308 residual=0.00032054216717369854
    DEBUG:root:i=309 residual=0.0003205435932613909
    DEBUG:root:i=310 residual=0.00031723958090879023
    DEBUG:root:i=311 residual=0.0003172382421325892
    DEBUG:root:i=312 residual=0.0003172381257172674
    DEBUG:root:i=313 residual=0.0003172380675096065
    DEBUG:root:i=314 residual=0.00031723949359729886
    DEBUG:root:i=315 residual=0.0003172380675096065
    DEBUG:root:i=316 residual=0.00032054094481281936
    DEBUG:root:i=317 residual=0.00032054222538135946
    DEBUG:root:i=318 residual=0.0003172410069964826
    DEBUG:root:i=319 residual=0.0003172381839249283
    DEBUG:root:i=320 residual=0.00031393542303703725
    DEBUG:root:i=321 residual=0.00031723672873340547
    DEBUG:root:i=322 residual=0.0003139354521408677
    DEBUG:root:i=323 residual=0.0003172366414219141
    DEBUG:root:i=324 residual=0.00031723672873340547
    DEBUG:root:i=325 residual=0.000317238038405776
    DEBUG:root:i=326 residual=0.00031723809661343694
    DEBUG:root:i=327 residual=0.00031393684912472963
    DEBUG:root:i=328 residual=0.0003172367869410664
    DEBUG:root:i=329 residual=0.000317238038405776
    DEBUG:root:i=330 residual=0.0003172381839249283
    DEBUG:root:i=331 residual=0.00032054074108600616
    DEBUG:root:i=332 residual=0.000320542196277529
    DEBUG:root:i=333 residual=0.0003172409487888217
    DEBUG:root:i=334 residual=0.00031723821302875876
    DEBUG:root:i=335 residual=0.0003172381839249283
    DEBUG:root:i=336 residual=0.0003172381257172674
    DEBUG:root:i=337 residual=0.0003172381257172674
    DEBUG:root:i=338 residual=0.0003172396100126207
    DEBUG:root:i=339 residual=0.0003172381257172674
    DEBUG:root:i=340 residual=0.0003172381257172674
    DEBUG:root:i=341 residual=0.0003172381257172674
    DEBUG:root:i=342 residual=0.0003172380675096065
    DEBUG:root:i=343 residual=0.00031723809661343694
    DEBUG:root:i=344 residual=0.0003172381257172674
    DEBUG:root:i=345 residual=0.00031723809661343694
    DEBUG:root:i=346 residual=0.00031723815482109785
    DEBUG:root:i=347 residual=0.0003172381257172674
    DEBUG:root:i=348 residual=0.00031723809661343694
    DEBUG:root:i=349 residual=0.000317238038405776
    DEBUG:root:i=350 residual=0.0003172381257172674
    DEBUG:root:i=351 residual=0.00031723949359729886
    DEBUG:root:i=352 residual=0.00031723821302875876
    DEBUG:root:i=353 residual=0.0003172381257172674
    DEBUG:root:i=354 residual=0.0003172380675096065
    DEBUG:root:i=355 residual=0.0003172381257172674
    DEBUG:root:i=356 residual=0.00031063274946063757
    DEBUG:root:i=357 residual=0.0003172354190610349
    DEBUG:root:i=358 residual=0.0003172380675096065
    DEBUG:root:i=359 residual=0.0003172381257172674
    DEBUG:root:i=360 residual=0.00031723809661343694
    DEBUG:root:i=361 residual=0.0003172381257172674
    DEBUG:root:i=362 residual=0.0003172381257172674
    DEBUG:root:i=363 residual=0.0003172381257172674
    DEBUG:root:i=364 residual=0.000320540857501328
    DEBUG:root:i=365 residual=0.00031723943538963795
    DEBUG:root:i=366 residual=0.0003172395518049598
    DEBUG:root:i=367 residual=0.0003172381257172674
    DEBUG:root:i=368 residual=0.0003172381257172674
    DEBUG:root:i=369 residual=0.00031723809661343694
    DEBUG:root:i=370 residual=0.00031723809661343694
    DEBUG:root:i=371 residual=0.0003172381257172674
    DEBUG:root:i=372 residual=0.0003172394644934684
    DEBUG:root:i=373 residual=0.0003139355103485286
    DEBUG:root:i=374 residual=0.00031723672873340547
    DEBUG:root:i=375 residual=0.0003172381257172674
    DEBUG:root:i=376 residual=0.00031723800930194557
    DEBUG:root:i=377 residual=0.0003172381839249283
    DEBUG:root:i=378 residual=0.0003172381257172674
    DEBUG:root:i=379 residual=0.0003172381839249283
    DEBUG:root:i=380 residual=0.0003172395227011293
    DEBUG:root:i=381 residual=0.0003172380675096065
    DEBUG:root:i=382 residual=0.00031723809661343694
    DEBUG:root:i=383 residual=0.00031723672873340547
    DEBUG:root:i=384 residual=0.00031723821302875876
    DEBUG:root:i=385 residual=0.00031723809661343694
    DEBUG:root:i=386 residual=0.0003172381257172674
    DEBUG:root:i=387 residual=0.00031393676181323826
    DEBUG:root:i=388 residual=0.00031723675783723593
    DEBUG:root:i=389 residual=0.000317238038405776
    DEBUG:root:i=390 residual=0.0003172381839249283
    DEBUG:root:i=391 residual=0.0003139353939332068
    DEBUG:root:i=392 residual=0.0003139339678455144
    DEBUG:root:i=393 residual=0.000317236699629575
    DEBUG:root:i=394 residual=0.0003139353939332068
    DEBUG:root:i=395 residual=0.00031063135247677565
    DEBUG:root:i=396 residual=0.00031393265817314386
    DEBUG:root:i=397 residual=0.0003106298972852528
    DEBUG:root:i=398 residual=0.00031393111567012966
    DEBUG:root:i=399 residual=0.00031393388053402305
    DEBUG:root:i=400 residual=0.00031723672873340547
    DEBUG:root:i=401 residual=0.000317238038405776
    DEBUG:root:i=402 residual=0.0003139354521408677
    DEBUG:root:i=403 residual=0.0003172366414219141
    DEBUG:root:i=404 residual=0.0003172381257172674
    DEBUG:root:i=405 residual=0.000317238038405776
    DEBUG:root:i=406 residual=0.0003172381257172674
    DEBUG:root:i=407 residual=0.0003172381839249283
    DEBUG:root:i=408 residual=0.0003172381257172674
    DEBUG:root:i=409 residual=0.00031723809661343694
    DEBUG:root:i=410 residual=0.0003172381839249283
    DEBUG:root:i=411 residual=0.0003172395518049598
    DEBUG:root:i=412 residual=0.00031723809661343694
    DEBUG:root:i=413 residual=0.0003139354521408677
    DEBUG:root:i=414 residual=0.00031393393874168396
    DEBUG:root:i=415 residual=0.0003139339678455144
    DEBUG:root:i=416 residual=0.000317236699629575
    DEBUG:root:i=417 residual=0.0003172381257172674
    DEBUG:root:i=418 residual=0.0003139353939332068
    DEBUG:root:i=419 residual=0.0003106313815806061
    DEBUG:root:i=420 residual=0.00031062847119756043
    DEBUG:root:i=421 residual=0.00030732579762116075
    DEBUG:root:i=422 residual=0.00031393111567012966
    DEBUG:root:i=423 residual=0.0003139339096378535
    DEBUG:root:i=424 residual=0.0003139339678455144
    DEBUG:root:i=425 residual=0.00031062980997376144
    DEBUG:root:i=426 residual=0.0003106285003013909
    DEBUG:root:i=427 residual=0.00031062986818142235
    DEBUG:root:i=428 residual=0.0003106298390775919
    DEBUG:root:i=429 residual=0.00031062852940522134
    DEBUG:root:i=430 residual=0.0003139324253425002
    DEBUG:root:i=431 residual=0.00031393254175782204
    DEBUG:root:i=432 residual=0.0003139339096378535
    DEBUG:root:i=433 residual=0.000317236699629575
    DEBUG:root:i=434 residual=0.00031723809661343694
    DEBUG:root:i=435 residual=0.0003172380675096065
    DEBUG:root:i=436 residual=0.0003172381257172674
    DEBUG:root:i=437 residual=0.00031723809661343694
    DEBUG:root:i=438 residual=0.0003172381257172674
    DEBUG:root:i=439 residual=0.00031723958090879023
    DEBUG:root:i=440 residual=0.0003172381257172674
    DEBUG:root:i=441 residual=0.00031393536482937634
    DEBUG:root:i=442 residual=0.0003139339678455144
    DEBUG:root:i=443 residual=0.0003073287080042064
    DEBUG:root:i=444 residual=0.00030732297454960644
    DEBUG:root:i=445 residual=0.00031062844209372997
    DEBUG:root:i=446 residual=0.0003139310283586383
    DEBUG:root:i=447 residual=0.0003106312360614538
    DEBUG:root:i=448 residual=0.0003073272528126836
    DEBUG:root:i=449 residual=0.000307322945445776
    DEBUG:root:i=450 residual=0.00031062838388606906
    DEBUG:root:i=451 residual=0.0003139324835501611
    DEBUG:root:i=452 residual=0.0003139339096378535
    DEBUG:root:i=453 residual=0.0003139325708616525
    DEBUG:root:i=454 residual=0.0003139325708616525
    DEBUG:root:i=455 residual=0.00031393393874168396
    DEBUG:root:i=456 residual=0.0003073286497965455
    DEBUG:root:i=457 residual=0.00030732579762116075
    DEBUG:root:i=458 residual=0.0003106271324213594
    DEBUG:root:i=459 residual=0.00031393245444633067
    DEBUG:root:i=460 residual=0.0003139325708616525
    DEBUG:root:i=461 residual=0.0003172366414219141
    DEBUG:root:i=462 residual=0.0003139353066217154
    DEBUG:root:i=463 residual=0.0003139340551570058
    DEBUG:root:i=464 residual=0.00031393393874168396
    DEBUG:root:i=465 residual=0.00031723675783723593
    DEBUG:root:i=466 residual=0.000317238038405776
    DEBUG:root:i=467 residual=0.000317238038405776
    DEBUG:root:i=468 residual=0.00031723809661343694
    DEBUG:root:i=469 residual=0.0003172381257172674
    DEBUG:root:i=470 residual=0.000317236699629575
    DEBUG:root:i=471 residual=0.0003172395518049598
    DEBUG:root:i=472 residual=0.000317238038405776
    DEBUG:root:i=473 residual=0.0003139354521408677
    DEBUG:root:i=474 residual=0.0003139339969493449
    DEBUG:root:i=475 residual=0.00031063135247677565
    DEBUG:root:i=476 residual=0.00031393117387779057
    DEBUG:root:i=477 residual=0.0003139339678455144
    DEBUG:root:i=478 residual=0.0003172365832142532
    DEBUG:root:i=479 residual=0.0003172380675096065
    DEBUG:root:i=480 residual=0.00031723809661343694
    DEBUG:root:i=481 residual=0.0003172381257172674
    DEBUG:root:i=482 residual=0.0003172381257172674
    DEBUG:root:i=483 residual=0.00031723815482109785
    DEBUG:root:i=484 residual=0.0003172381257172674
    DEBUG:root:i=485 residual=0.0003172381257172674
    DEBUG:root:i=486 residual=0.0003172381839249283
    DEBUG:root:i=487 residual=0.0003172394644934684
    DEBUG:root:i=488 residual=0.0003172381839249283
    DEBUG:root:i=489 residual=0.000317238038405776
    DEBUG:root:i=490 residual=0.0003172381257172674
    DEBUG:root:i=491 residual=0.0003205407992936671
    DEBUG:root:i=492 residual=0.0003172409487888217
    DEBUG:root:i=493 residual=0.0003172381839249283
    DEBUG:root:i=494 residual=0.0003172380675096065
    DEBUG:root:i=495 residual=0.0003139354521408677
    DEBUG:root:i=496 residual=0.000317236699629575
    DEBUG:root:i=497 residual=0.0003172381257172674
    DEBUG:root:i=498 residual=0.0003172380675096065
    DEBUG:root:i=499 residual=0.00032054088660515845
    DEBUG:root:i=500 residual=0.00031724083237349987
    DEBUG:root:i=501 residual=0.00031723821302875876
    DEBUG:root:i=502 residual=0.00031723809661343694
    DEBUG:root:i=503 residual=0.00031723675783723593
    DEBUG:root:i=504 residual=0.0003139355103485286
    DEBUG:root:i=505 residual=0.0003172366414219141
    DEBUG:root:i=506 residual=0.00031723958090879023
    DEBUG:root:i=507 residual=0.0003172381257172674
    DEBUG:root:i=508 residual=0.0003139354521408677
    DEBUG:root:i=509 residual=0.0003139339678455144
    DEBUG:root:i=510 residual=0.00031063135247677565
    DEBUG:root:i=511 residual=0.00031723390566185117
    DEBUG:root:i=512 residual=0.0003172394062858075
    DEBUG:root:i=513 residual=0.00031723809661343694
    DEBUG:root:i=514 residual=0.0003172381257172674
    DEBUG:root:i=515 residual=0.0003172380675096065
    DEBUG:root:i=516 residual=0.00031723672873340547
    DEBUG:root:i=517 residual=0.000317238038405776
    DEBUG:root:i=518 residual=0.0003172381839249283
    DEBUG:root:i=519 residual=0.0003172395518049598
    DEBUG:root:i=520 residual=0.0003172381257172674
    DEBUG:root:i=521 residual=0.0003106328076682985
    DEBUG:root:i=522 residual=0.0003106298390775919
    DEBUG:root:i=523 residual=0.00031062852940522134
    DEBUG:root:i=524 residual=0.0003139325126539916
    DEBUG:root:i=525 residual=0.0003139339678455144
    DEBUG:root:i=526 residual=0.00031723527354188263
    DEBUG:root:i=527 residual=0.0003172379801981151
    DEBUG:root:i=528 residual=0.00031723809661343694
    DEBUG:root:i=529 residual=0.0003172380675096065
    DEBUG:root:i=530 residual=0.00031723815482109785
    DEBUG:root:i=531 residual=0.00031393690733239055
    DEBUG:root:i=532 residual=0.00031393408426083624
    DEBUG:root:i=533 residual=0.0003139326290693134
    DEBUG:root:i=534 residual=0.0003139339678455144
    DEBUG:root:i=535 residual=0.0003139339969493449
    DEBUG:root:i=536 residual=0.00031063129426911473
    DEBUG:root:i=537 residual=0.00031062852940522134
    DEBUG:root:i=538 residual=0.0003106297517661005
    DEBUG:root:i=539 residual=0.00031393111567012966
    DEBUG:root:i=540 residual=0.0003139339678455144
    DEBUG:root:i=541 residual=0.00031063129426911473
    DEBUG:root:i=542 residual=0.00031393259996548295
    DEBUG:root:i=543 residual=0.00031393245444633067
    DEBUG:root:i=544 residual=0.0003106312651652843
    DEBUG:root:i=545 residual=0.00031062852940522134
    DEBUG:root:i=546 residual=0.00031062980997376144
    DEBUG:root:i=547 residual=0.00031062980997376144
    DEBUG:root:i=548 residual=0.00031062847119756043
    DEBUG:root:i=549 residual=0.00031393105746246874
    DEBUG:root:i=550 residual=0.0003073286497965455
    DEBUG:root:i=551 residual=0.0003106284129898995
    DEBUG:root:i=552 residual=0.00031062847119756043
    DEBUG:root:i=553 residual=0.0003106298390775919
    DEBUG:root:i=554 residual=0.00031062838388606906
    DEBUG:root:i=555 residual=0.000313931202981621
    DEBUG:root:i=556 residual=0.0003172366414219141
    DEBUG:root:i=557 residual=0.0003172380675096065
    DEBUG:root:i=558 residual=0.0003172395227011293
    DEBUG:root:i=559 residual=0.00031723809661343694
    DEBUG:root:i=560 residual=0.0003172381257172674
    DEBUG:root:i=561 residual=0.0003172381257172674
    DEBUG:root:i=562 residual=0.0003172380675096065
    DEBUG:root:i=563 residual=0.00031723815482109785
    DEBUG:root:i=564 residual=0.000317238038405776
    DEBUG:root:i=565 residual=0.0003106328076682985
    DEBUG:root:i=566 residual=0.0003106285585090518
    DEBUG:root:i=567 residual=0.0003073272528126836
    DEBUG:root:i=568 residual=0.00031392977689392865
    DEBUG:root:i=569 residual=0.0003139339678455144
    DEBUG:root:i=570 residual=0.0003139338514301926
    DEBUG:root:i=571 residual=0.0003106313233729452
    DEBUG:root:i=572 residual=0.0003106285003013909
    DEBUG:root:i=573 residual=0.00031393254175782204
    DEBUG:root:i=574 residual=0.0003139339678455144
    DEBUG:root:i=575 residual=0.00030732713639736176
    DEBUG:root:i=576 residual=0.00031393117387779057
    DEBUG:root:i=577 residual=0.00031723655411042273
    DEBUG:root:i=578 residual=0.0003139339678455144
    DEBUG:root:i=579 residual=0.00031063135247677565
    DEBUG:root:i=580 residual=0.0003139325708616525
    DEBUG:root:i=581 residual=0.00031063135247677565
    DEBUG:root:i=582 residual=0.00031062844209372997
    DEBUG:root:i=583 residual=0.00031062992638908327
    DEBUG:root:i=584 residual=0.0003139325126539916
    DEBUG:root:i=585 residual=0.00031393254175782204
    DEBUG:root:i=586 residual=0.00031393254175782204
    DEBUG:root:i=587 residual=0.00031723661231808364
    DEBUG:root:i=588 residual=0.0003172395227011293
    DEBUG:root:i=589 residual=0.00031723809661343694
    DEBUG:root:i=590 residual=0.0003172381257172674
    DEBUG:root:i=591 residual=0.0003172381257172674
    DEBUG:root:i=592 residual=0.0003172381257172674
    DEBUG:root:i=593 residual=0.00031723809661343694
    DEBUG:root:i=594 residual=0.0003139355103485286
    DEBUG:root:i=595 residual=0.0003139340551570058
    DEBUG:root:i=596 residual=0.00031723518623039126
    DEBUG:root:i=597 residual=0.0003139353939332068
    DEBUG:root:i=598 residual=0.00030732867890037596
    DEBUG:root:i=599 residual=0.00031393117387779057
    DEBUG:root:i=600 residual=0.0003106312651652843
    DEBUG:root:i=601 residual=0.00031393117387779057
    DEBUG:root:i=602 residual=0.00031723661231808364
    DEBUG:root:i=603 residual=0.00031723800930194557
    DEBUG:root:i=604 residual=0.0003139354521408677
    DEBUG:root:i=605 residual=0.0003172367869410664
    DEBUG:root:i=606 residual=0.0003172381257172674
    DEBUG:root:i=607 residual=0.000317238038405776
    DEBUG:root:i=608 residual=0.0003139354521408677
    DEBUG:root:i=609 residual=0.00031063135247677565
    DEBUG:root:i=610 residual=0.0003139326290693134
    DEBUG:root:i=611 residual=0.0003139339678455144
    DEBUG:root:i=612 residual=0.00031063129426911473
    DEBUG:root:i=613 residual=0.0003139311447739601
    DEBUG:root:i=614 residual=0.0003073272237088531
    DEBUG:root:i=615 residual=0.00031062844209372997
    DEBUG:root:i=616 residual=0.00031062844209372997
    DEBUG:root:i=617 residual=0.00031393254175782204
    DEBUG:root:i=618 residual=0.0003073285333812237
    DEBUG:root:i=619 residual=0.0003139298059977591
    DEBUG:root:i=620 residual=0.00031062986818142235
    DEBUG:root:i=621 residual=0.0003205379471182823
    DEBUG:root:i=622 residual=0.0003172394644934684
    DEBUG:root:i=623 residual=0.00031723943538963795
    DEBUG:root:i=624 residual=0.0003139354521408677
    DEBUG:root:i=625 residual=0.0003139340551570058
    DEBUG:root:i=626 residual=0.00031723672873340547
    DEBUG:root:i=627 residual=0.00031723800930194557
    DEBUG:root:i=628 residual=0.00031723809661343694
    DEBUG:root:i=629 residual=0.0003172381257172674
    DEBUG:root:i=630 residual=0.00031723672873340547
    DEBUG:root:i=631 residual=0.00031723809661343694
    DEBUG:root:i=632 residual=0.0003172381257172674
    DEBUG:root:i=633 residual=0.0003172395227011293
    DEBUG:root:i=634 residual=0.00031723809661343694
    DEBUG:root:i=635 residual=0.0003172381257172674
    DEBUG:root:i=636 residual=0.000317238038405776
    DEBUG:root:i=637 residual=0.0003172381839249283
    DEBUG:root:i=638 residual=0.0003172380675096065
    DEBUG:root:i=639 residual=0.0003172395227011293
    DEBUG:root:i=640 residual=0.0003139354521408677
    DEBUG:root:i=641 residual=0.0003172367869410664
    DEBUG:root:i=642 residual=0.0003172381257172674
    DEBUG:root:i=643 residual=0.00031723809661343694
    DEBUG:root:i=644 residual=0.0003139354521408677
    DEBUG:root:i=645 residual=0.0003106312651652843
    DEBUG:root:i=646 residual=0.0003139325708616525
    DEBUG:root:i=647 residual=0.0003139325708616525
    DEBUG:root:i=648 residual=0.0003106312651652843
    DEBUG:root:i=649 residual=0.00031393117387779057
    DEBUG:root:i=650 residual=0.0003139346372336149
    DEBUG:root:i=651 residual=0.00031723707797937095
    DEBUG:root:i=652 residual=0.0003172383294440806
    DEBUG:root:i=653 residual=0.00031393548124469817
    DEBUG:root:i=654 residual=0.0003172354190610349
    DEBUG:root:i=655 residual=0.0003172381257172674
    DEBUG:root:i=656 residual=0.0003172395227011293
    DEBUG:root:i=657 residual=0.00031723809661343694
    DEBUG:root:i=658 residual=0.00031723809661343694
    DEBUG:root:i=659 residual=0.0003139354521408677
    DEBUG:root:i=660 residual=0.00031393402605317533
    DEBUG:root:i=661 residual=0.00031723672873340547
    DEBUG:root:i=662 residual=0.0003172379801981151
    DEBUG:root:i=663 residual=0.0003172381257172674
    DEBUG:root:i=664 residual=0.0003172381257172674
    DEBUG:root:i=665 residual=0.0003172395227011293
    DEBUG:root:i=666 residual=0.00031723815482109785
    DEBUG:root:i=667 residual=0.0003172381257172674
    DEBUG:root:i=668 residual=0.00031723809661343694
    DEBUG:root:i=669 residual=0.00031723809661343694
    DEBUG:root:i=670 residual=0.0003172381257172674
    DEBUG:root:i=671 residual=0.00031723672873340547
    DEBUG:root:i=672 residual=0.00031723949359729886
    DEBUG:root:i=673 residual=0.00031723809661343694
    DEBUG:root:i=674 residual=0.00031723809661343694
    DEBUG:root:i=675 residual=0.0003172381257172674
    DEBUG:root:i=676 residual=0.0003139354521408677
    DEBUG:root:i=677 residual=0.00031393408426083624
    DEBUG:root:i=678 residual=0.00031393408426083624
    DEBUG:root:i=679 residual=0.00031393393874168396
    DEBUG:root:i=680 residual=0.00031063135247677565
    DEBUG:root:i=681 residual=0.00031062986818142235
    DEBUG:root:i=682 residual=0.00031392983510158956
    DEBUG:root:i=683 residual=0.0003139339096378535
    DEBUG:root:i=684 residual=0.00031393393874168396
    DEBUG:root:i=685 residual=0.0003172366414219141
    DEBUG:root:i=686 residual=0.0003172380675096065
    DEBUG:root:i=687 residual=0.00031723815482109785
    DEBUG:root:i=688 residual=0.00031393684912472963
    DEBUG:root:i=689 residual=0.000317235360853374
    DEBUG:root:i=690 residual=0.00031723943538963795
    DEBUG:root:i=691 residual=0.0003172381257172674
    DEBUG:root:i=692 residual=0.0003172381257172674
    DEBUG:root:i=693 residual=0.00031723809661343694
    DEBUG:root:i=694 residual=0.00031723672873340547
    DEBUG:root:i=695 residual=0.00031723800930194557
    DEBUG:root:i=696 residual=0.0003172395518049598
    DEBUG:root:i=697 residual=0.0003172381257172674
    DEBUG:root:i=698 residual=0.0003172381257172674
    DEBUG:root:i=699 residual=0.0003172380675096065
    DEBUG:root:i=700 residual=0.00031723815482109785
    DEBUG:root:i=701 residual=0.00031723809661343694
    DEBUG:root:i=702 residual=0.00031723809661343694
    DEBUG:root:i=703 residual=0.0003172395518049598
    DEBUG:root:i=704 residual=0.00031723815482109785
    DEBUG:root:i=705 residual=0.00031723821302875876
    DEBUG:root:i=706 residual=0.000317238038405776
    DEBUG:root:i=707 residual=0.0003172381257172674
    DEBUG:root:i=708 residual=0.0003172381257172674
    DEBUG:root:i=709 residual=0.00031723949359729886
    DEBUG:root:i=710 residual=0.00031723815482109785
    DEBUG:root:i=711 residual=0.0003139354521408677
    DEBUG:root:i=712 residual=0.0003139339678455144
    DEBUG:root:i=713 residual=0.00031723527354188263
    DEBUG:root:i=714 residual=0.00031393542303703725
    DEBUG:root:i=715 residual=0.00031723667052574456
    DEBUG:root:i=716 residual=0.00031723809661343694
    DEBUG:root:i=717 residual=0.000317238038405776
    DEBUG:root:i=718 residual=0.00031723809661343694
    DEBUG:root:i=719 residual=0.00031723815482109785
    DEBUG:root:i=720 residual=0.0003172395518049598
    DEBUG:root:i=721 residual=0.0003172381257172674
    DEBUG:root:i=722 residual=0.0003172381839249283
    DEBUG:root:i=723 residual=0.000317238038405776
    DEBUG:root:i=724 residual=0.0003172381257172674
    DEBUG:root:i=725 residual=0.0003172381257172674
    DEBUG:root:i=726 residual=0.00031723809661343694
    DEBUG:root:i=727 residual=0.0003172395518049598
    DEBUG:root:i=728 residual=0.0003172381257172674
    DEBUG:root:i=729 residual=0.0003172381257172674
    DEBUG:root:i=730 residual=0.00031393402605317533
    DEBUG:root:i=731 residual=0.00031723672873340547
    DEBUG:root:i=732 residual=0.000317238038405776
    DEBUG:root:i=733 residual=0.00031723815482109785
    DEBUG:root:i=734 residual=0.00031723800930194557
    DEBUG:root:i=735 residual=0.00031723949359729886
    DEBUG:root:i=736 residual=0.00031723815482109785
    DEBUG:root:i=737 residual=0.0003172381257172674
    DEBUG:root:i=738 residual=0.00031723815482109785
    DEBUG:root:i=739 residual=0.00031723800930194557
    DEBUG:root:i=740 residual=0.0003172381257172674
    DEBUG:root:i=741 residual=0.00031723949359729886
    DEBUG:root:i=742 residual=0.0003172381839249283
    DEBUG:root:i=743 residual=0.0003172381257172674
    DEBUG:root:i=744 residual=0.0003106328076682985
    DEBUG:root:i=745 residual=0.00031393259996548295
    DEBUG:root:i=746 residual=0.0003139325708616525
    DEBUG:root:i=747 residual=0.00031063129426911473
    DEBUG:root:i=748 residual=0.000313931202981621
    DEBUG:root:i=749 residual=0.0003139339096378535
    DEBUG:root:i=750 residual=0.00031723667052574456
    DEBUG:root:i=751 residual=0.000317238038405776
    DEBUG:root:i=752 residual=0.00032054082839749753
    DEBUG:root:i=753 residual=0.0003172408905811608
    DEBUG:root:i=754 residual=0.0003172382712364197
    DEBUG:root:i=755 residual=0.0003172381257172674
    DEBUG:root:i=756 residual=0.000317238038405776
    DEBUG:root:i=757 residual=0.0003172381257172674
    DEBUG:root:i=758 residual=0.0003172381257172674
    DEBUG:root:i=759 residual=0.0003172381257172674
    DEBUG:root:i=760 residual=0.00031393542303703725
    DEBUG:root:i=761 residual=0.0003172367869410664
    DEBUG:root:i=762 residual=0.0003172379801981151
    DEBUG:root:i=763 residual=0.0003172381257172674
    DEBUG:root:i=764 residual=0.0003139354521408677
    DEBUG:root:i=765 residual=0.00031393408426083624
    DEBUG:root:i=766 residual=0.0003139339969493449
    DEBUG:root:i=767 residual=0.0003172366414219141
    DEBUG:root:i=768 residual=0.000317238038405776
    DEBUG:root:i=769 residual=0.0003172381257172674
    DEBUG:root:i=770 residual=0.00031723809661343694
    DEBUG:root:i=771 residual=0.0003172381257172674
    DEBUG:root:i=772 residual=0.00031723809661343694
    DEBUG:root:i=773 residual=0.0003139353939332068
    DEBUG:root:i=774 residual=0.00031393402605317533
    DEBUG:root:i=775 residual=0.00031063135247677565
    DEBUG:root:i=776 residual=0.00031393131939694285
    DEBUG:root:i=777 residual=0.00031393388053402305
    DEBUG:root:i=778 residual=0.0003172366414219141
    DEBUG:root:i=779 residual=0.0003172380675096065
    DEBUG:root:i=780 residual=0.00031723949359729886
    DEBUG:root:i=781 residual=0.00031393548124469817
    DEBUG:root:i=782 residual=0.00031723672873340547
    DEBUG:root:i=783 residual=0.0003205393149983138
    DEBUG:root:i=784 residual=0.00031723949359729886
    DEBUG:root:i=785 residual=0.0003172381257172674
    DEBUG:root:i=786 residual=0.00031723958090879023
    DEBUG:root:i=787 residual=0.0003172381257172674
    DEBUG:root:i=788 residual=0.00031723815482109785
    DEBUG:root:i=789 residual=0.00031723809661343694
    DEBUG:root:i=790 residual=0.00031393542303703725
    DEBUG:root:i=791 residual=0.00031393402605317533
    DEBUG:root:i=792 residual=0.000317236699629575
    DEBUG:root:i=793 residual=0.00031723809661343694
    DEBUG:root:i=794 residual=0.0003172381257172674
    DEBUG:root:i=795 residual=0.0003172381257172674
    DEBUG:root:i=796 residual=0.0003172394644934684
    DEBUG:root:i=797 residual=0.0003172381839249283
    DEBUG:root:i=798 residual=0.0003172380675096065
    DEBUG:root:i=799 residual=0.0003172381257172674
    DEBUG:root:i=800 residual=0.0003172380675096065
    DEBUG:root:i=801 residual=0.0003172367869410664
    DEBUG:root:i=802 residual=0.0003172395227011293
    DEBUG:root:i=803 residual=0.0003172381257172674
    DEBUG:root:i=804 residual=0.00031723815482109785
    DEBUG:root:i=805 residual=0.0003172380675096065
    DEBUG:root:i=806 residual=0.00031393542303703725
    DEBUG:root:i=807 residual=0.00031723667052574456
    DEBUG:root:i=808 residual=0.0003205407992936671
    DEBUG:root:i=809 residual=0.0003172408905811608
    DEBUG:root:i=810 residual=0.0003172382421325892
    DEBUG:root:i=811 residual=0.0003172381257172674
    DEBUG:root:i=812 residual=0.00031723809661343694
    DEBUG:root:i=813 residual=0.00031723815482109785
    DEBUG:root:i=814 residual=0.0003172381257172674
    DEBUG:root:i=815 residual=0.0003172395227011293
    DEBUG:root:i=816 residual=0.0003172381257172674
    DEBUG:root:i=817 residual=0.0003172380675096065
    DEBUG:root:i=818 residual=0.00031723809661343694
    DEBUG:root:i=819 residual=0.0003172381839249283
    DEBUG:root:i=820 residual=0.0003172381257172674
    DEBUG:root:i=821 residual=0.00031723672873340547
    DEBUG:root:i=822 residual=0.00031723943538963795
    DEBUG:root:i=823 residual=0.0003172381257172674
    DEBUG:root:i=824 residual=0.00031723815482109785
    DEBUG:root:i=825 residual=0.0003139353939332068
    DEBUG:root:i=826 residual=0.0003172367869410664
    DEBUG:root:i=827 residual=0.000317238038405776
    DEBUG:root:i=828 residual=0.000317238038405776
    DEBUG:root:i=829 residual=0.00031723809661343694
    DEBUG:root:i=830 residual=0.00031723958090879023
    DEBUG:root:i=831 residual=0.00031723815482109785
    DEBUG:root:i=832 residual=0.0003172381839249283
    DEBUG:root:i=833 residual=0.00031723809661343694
    DEBUG:root:i=834 residual=0.00031723809661343694
    DEBUG:root:i=835 residual=0.00031723815482109785
    DEBUG:root:i=836 residual=0.0003172394644934684
    DEBUG:root:i=837 residual=0.0003172381839249283
    DEBUG:root:i=838 residual=0.00031393542303703725
    DEBUG:root:i=839 residual=0.0003106298972852528
    DEBUG:root:i=840 residual=0.00031723390566185117
    DEBUG:root:i=841 residual=0.00031723943538963795
    DEBUG:root:i=842 residual=0.00031723809661343694
    DEBUG:root:i=843 residual=0.0003172381839249283
    DEBUG:root:i=844 residual=0.0003172380675096065
    DEBUG:root:i=845 residual=0.0003172381257172674
    DEBUG:root:i=846 residual=0.0003172381257172674
    DEBUG:root:i=847 residual=0.0003172395227011293
    DEBUG:root:i=848 residual=0.0003172381257172674
    DEBUG:root:i=849 residual=0.0003172381257172674
    DEBUG:root:i=850 residual=0.00031723809661343694
    DEBUG:root:i=851 residual=0.0003172380675096065
    DEBUG:root:i=852 residual=0.0003172381257172674
    DEBUG:root:i=853 residual=0.00031723815482109785
    DEBUG:root:i=854 residual=0.0003172381257172674
    DEBUG:root:i=855 residual=0.0003172381257172674
    DEBUG:root:i=856 residual=0.000317238038405776
    DEBUG:root:i=857 residual=0.0003172381257172674
    DEBUG:root:i=858 residual=0.0003172381257172674
    DEBUG:root:i=859 residual=0.00031723815482109785
    DEBUG:root:i=860 residual=0.00031723949359729886
    DEBUG:root:i=861 residual=0.00031723809661343694
    DEBUG:root:i=862 residual=0.0003172381257172674
    DEBUG:root:i=863 residual=0.00031723815482109785
    DEBUG:root:i=864 residual=0.0003139353939332068
    DEBUG:root:i=865 residual=0.00031393402605317533
    DEBUG:root:i=866 residual=0.00031063135247677565
    DEBUG:root:i=867 residual=0.00031723527354188263
    DEBUG:root:i=868 residual=0.000317238038405776
    DEBUG:root:i=869 residual=0.00031393542303703725
    DEBUG:root:i=870 residual=0.0003073272528126836
    DEBUG:root:i=871 residual=0.00031392986420542
    DEBUG:root:i=872 residual=0.00032053925679065287
    DEBUG:root:i=873 residual=0.0003172408614773303
    DEBUG:root:i=874 residual=0.0003172381257172674
    DEBUG:root:i=875 residual=0.00031723815482109785
    DEBUG:root:i=876 residual=0.0003172381257172674
    DEBUG:root:i=877 residual=0.0003172381257172674
    DEBUG:root:i=878 residual=0.000317238038405776
    DEBUG:root:i=879 residual=0.0003172395518049598
    DEBUG:root:i=880 residual=0.00031393548124469817
    DEBUG:root:i=881 residual=0.00031723672873340547
    DEBUG:root:i=882 residual=0.0003172381257172674
    DEBUG:root:i=883 residual=0.0003172366414219141
    DEBUG:root:i=884 residual=0.0003172381257172674
    DEBUG:root:i=885 residual=0.0003172380675096065
    DEBUG:root:i=886 residual=0.00031723815482109785
    DEBUG:root:i=887 residual=0.00031723949359729886
    DEBUG:root:i=888 residual=0.00031723815482109785
    DEBUG:root:i=889 residual=0.00031723809661343694
    DEBUG:root:i=890 residual=0.0003172381257172674
    DEBUG:root:i=891 residual=0.0003172381257172674
    DEBUG:root:i=892 residual=0.00031723809661343694
    DEBUG:root:i=893 residual=0.000313936936436221
    DEBUG:root:i=894 residual=0.00031062986818142235
    DEBUG:root:i=895 residual=0.0003139325708616525
    DEBUG:root:i=896 residual=0.0003106312651652843
    DEBUG:root:i=897 residual=0.00031062847119756043
    DEBUG:root:i=898 residual=0.00030732579762116075
    DEBUG:root:i=899 residual=0.0003139310865662992
    DEBUG:root:i=900 residual=0.00031723661231808364
    DEBUG:root:i=901 residual=0.00031393536482937634
    DEBUG:root:i=902 residual=0.0003139339969493449
    DEBUG:root:i=903 residual=0.0003172353026457131
    DEBUG:root:i=904 residual=0.0003172380675096065
    DEBUG:root:i=905 residual=0.00031393536482937634
    DEBUG:root:i=906 residual=0.000317236699629575
    DEBUG:root:i=907 residual=0.0003172380675096065
    DEBUG:root:i=908 residual=0.0003172381257172674
    DEBUG:root:i=909 residual=0.0003172395227011293
    DEBUG:root:i=910 residual=0.0003172381257172674
    DEBUG:root:i=911 residual=0.00031723809661343694
    DEBUG:root:i=912 residual=0.0003139355103485286
    DEBUG:root:i=913 residual=0.00031393402605317533
    DEBUG:root:i=914 residual=0.0003106313233729452
    DEBUG:root:i=915 residual=0.0003139325708616525
    DEBUG:root:i=916 residual=0.00031062847119756043
    DEBUG:root:i=917 residual=0.0003139325126539916
    DEBUG:root:i=918 residual=0.0003139339969493449
    DEBUG:root:i=919 residual=0.0003172366414219141
    DEBUG:root:i=920 residual=0.00031723809661343694
    DEBUG:root:i=921 residual=0.0003172380675096065
    DEBUG:root:i=922 residual=0.00031723809661343694
    DEBUG:root:i=923 residual=0.0003172395227011293
    DEBUG:root:i=924 residual=0.0003172381839249283
    DEBUG:root:i=925 residual=0.00031723809661343694
    DEBUG:root:i=926 residual=0.00031723667052574456
    DEBUG:root:i=927 residual=0.000317238038405776
    DEBUG:root:i=928 residual=0.00031393548124469817
    DEBUG:root:i=929 residual=0.00031393408426083624
    DEBUG:root:i=930 residual=0.00031063135247677565
    DEBUG:root:i=931 residual=0.0003073272528126836
    DEBUG:root:i=932 residual=0.00031062704510986805
    DEBUG:root:i=933 residual=0.0003172352153342217
    DEBUG:root:i=934 residual=0.0003172366414219141
    DEBUG:root:i=935 residual=0.00031723809661343694
    DEBUG:root:i=936 residual=0.0003172395227011293
    DEBUG:root:i=937 residual=0.0003205407701898366
    DEBUG:root:i=938 residual=0.0003172395227011293
    DEBUG:root:i=939 residual=0.0003172381257172674
    DEBUG:root:i=940 residual=0.0003139354521408677
    DEBUG:root:i=941 residual=0.000317236699629575
    DEBUG:root:i=942 residual=0.0003172381257172674
    DEBUG:root:i=943 residual=0.0003172394062858075
    DEBUG:root:i=944 residual=0.0003172381839249283
    DEBUG:root:i=945 residual=0.0003172380675096065
    DEBUG:root:i=946 residual=0.0003172382421325892
    DEBUG:root:i=947 residual=0.0003172381257172674
    DEBUG:root:i=948 residual=0.000317238038405776
    DEBUG:root:i=949 residual=0.0003172395227011293
    DEBUG:root:i=950 residual=0.0003172381257172674
    DEBUG:root:i=951 residual=0.0003172381257172674
    DEBUG:root:i=952 residual=0.00031723809661343694
    DEBUG:root:i=953 residual=0.0003172367869410664
    DEBUG:root:i=954 residual=0.00031723800930194557
    DEBUG:root:i=955 residual=0.00031723815482109785
    DEBUG:root:i=956 residual=0.0003172395227011293
    DEBUG:root:i=957 residual=0.0003139355103485286
    DEBUG:root:i=958 residual=0.000317236699629575
    DEBUG:root:i=959 residual=0.0003139353357255459
    DEBUG:root:i=960 residual=0.000317236699629575
    DEBUG:root:i=961 residual=0.0003139353939332068
    DEBUG:root:i=962 residual=0.0003205394314136356
    DEBUG:root:i=963 residual=0.0003172395518049598
    DEBUG:root:i=964 residual=0.00031723815482109785
    DEBUG:root:i=965 residual=0.00031723943538963795
    DEBUG:root:i=966 residual=0.0003172381839249283
    DEBUG:root:i=967 residual=0.0003205407992936671
    DEBUG:root:i=968 residual=0.000320542196277529
    DEBUG:root:i=969 residual=0.0003172409487888217
    DEBUG:root:i=970 residual=0.00031723815482109785
    DEBUG:root:i=971 residual=0.00031723815482109785
    DEBUG:root:i=972 residual=0.0003172380675096065
    DEBUG:root:i=973 residual=0.0003172367869410664
    DEBUG:root:i=974 residual=0.000317238038405776
    DEBUG:root:i=975 residual=0.0003172395518049598
    DEBUG:root:i=976 residual=0.0003139353939332068
    DEBUG:root:i=977 residual=0.0003172367869410664
    DEBUG:root:i=978 residual=0.00031723809661343694
    DEBUG:root:i=979 residual=0.0003172381257172674
    DEBUG:root:i=980 residual=0.00031723815482109785
    DEBUG:root:i=981 residual=0.0003172380675096065
    DEBUG:root:i=982 residual=0.0003172380675096065
    DEBUG:root:i=983 residual=0.00031723949359729886
    DEBUG:root:i=984 residual=0.00031723815482109785
    DEBUG:root:i=985 residual=0.00031723815482109785
    DEBUG:root:i=986 residual=0.00031723815482109785
    DEBUG:root:i=987 residual=0.0003172380675096065
    DEBUG:root:i=988 residual=0.00031723809661343694
    DEBUG:root:i=989 residual=0.0003172381257172674
    DEBUG:root:i=990 residual=0.00031723958090879023
    DEBUG:root:i=991 residual=0.0003172381257172674
    DEBUG:root:i=992 residual=0.000317236699629575
    DEBUG:root:i=993 residual=0.000317238038405776
    DEBUG:root:i=994 residual=0.00031723809661343694
    DEBUG:root:i=995 residual=0.00031723841675557196
    DEBUG:root:i=996 residual=0.0003172396100126207
    DEBUG:root:i=997 residual=0.0003172381839249283
    DEBUG:root:i=998 residual=0.0003172381257172674
    DEBUG:root:i=999 residual=0.00032054074108600616
    INFO:root:rank=0 pagerank=1.0624e+01 url=www.lawfareblog.com/snowden-revelations
    INFO:root:rank=1 pagerank=1.0624e+01 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=1.0624e+01 url=www.lawfareblog.com/masthead
    INFO:root:rank=3 pagerank=1.0624e+01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=1.0624e+01 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=1.0624e+01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=6 pagerank=1.0624e+01 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=7 pagerank=1.0624e+01 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=1.0624e+01 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=1.0624e+01 url=www.lawfareblog.com/topics
  
  
  
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
   DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=5.117094993591309
    DEBUG:root:i=1 residual=2.980546712875366
    DEBUG:root:i=2 residual=2.44927978515625
    DEBUG:root:i=3 residual=1.6987189054489136
    DEBUG:root:i=4 residual=1.1001710891723633
    DEBUG:root:i=5 residual=0.7298129796981812
    DEBUG:root:i=6 residual=0.5148047804832458
    DEBUG:root:i=7 residual=0.37960749864578247
    DEBUG:root:i=8 residual=0.285101056098938
    DEBUG:root:i=9 residual=0.21571314334869385
    DEBUG:root:i=10 residual=0.16483232378959656
    DEBUG:root:i=11 residual=0.12841448187828064
    DEBUG:root:i=12 residual=0.10304442048072815
    DEBUG:root:i=13 residual=0.08559015393257141
    DEBUG:root:i=14 residual=0.07334044575691223
    DEBUG:root:i=15 residual=0.0642278864979744
    DEBUG:root:i=16 residual=0.05689682066440582
    DEBUG:root:i=17 residual=0.05056967958807945
    DEBUG:root:i=18 residual=0.044862281531095505
    DEBUG:root:i=19 residual=0.03961113467812538
    DEBUG:root:i=20 residual=0.03476466238498688
    DEBUG:root:i=21 residual=0.0303158201277256
    DEBUG:root:i=22 residual=0.026269709691405296
    DEBUG:root:i=23 residual=0.022627854719758034
    DEBUG:root:i=24 residual=0.01938468962907791
    DEBUG:root:i=25 residual=0.016523731872439384
    DEBUG:root:i=26 residual=0.01402216125279665
    DEBUG:root:i=27 residual=0.011851786635816097
    DEBUG:root:i=28 residual=0.00998249277472496
    DEBUG:root:i=29 residual=0.00838180910795927
    DEBUG:root:i=30 residual=0.007018709089607
    DEBUG:root:i=31 residual=0.00586334802210331
    DEBUG:root:i=32 residual=0.004888011142611504
    DEBUG:root:i=33 residual=0.0040672835893929005
    DEBUG:root:i=34 residual=0.003379765897989273
    DEBUG:root:i=35 residual=0.0028046041261404753
    DEBUG:root:i=36 residual=0.0023247161880135536
    DEBUG:root:i=37 residual=0.0019252862548455596
    DEBUG:root:i=38 residual=0.0015933752292767167
    DEBUG:root:i=39 residual=0.0013178696390241385
    DEBUG:root:i=40 residual=0.0010894584702327847
    DEBUG:root:i=41 residual=0.0009004308376461267
    DEBUG:root:i=42 residual=0.0007439177134074271
    DEBUG:root:i=43 residual=0.0006144192302599549
    DEBUG:root:i=44 residual=0.0005076350062154233
    DEBUG:root:i=45 residual=0.00041931981104426086
    DEBUG:root:i=46 residual=0.0003465200716163963
    DEBUG:root:i=47 residual=0.0002862656256183982
    DEBUG:root:i=48 residual=0.00023636473633814603
    DEBUG:root:i=49 residual=0.00019551723380573094
    DEBUG:root:i=50 residual=0.00016160412633325905
    DEBUG:root:i=51 residual=0.00013361302262637764
    DEBUG:root:i=52 residual=0.00011055950744776055
    DEBUG:root:i=53 residual=9.150529513135552e-05
    DEBUG:root:i=54 residual=7.570399611722678e-05
    DEBUG:root:i=55 residual=6.26765686320141e-05
    DEBUG:root:i=56 residual=5.201449675951153e-05
    DEBUG:root:i=57 residual=4.297796112950891e-05
    DEBUG:root:i=58 residual=3.558820753823966e-05
    DEBUG:root:i=59 residual=2.9582239221781492e-05
    DEBUG:root:i=60 residual=2.4519797079847194e-05
    DEBUG:root:i=61 residual=2.0358769688755274e-05
    DEBUG:root:i=62 residual=1.6953244994510897e-05
    DEBUG:root:i=63 residual=1.4061471119930502e-05
    DEBUG:root:i=64 residual=1.1682228432619013e-05
    DEBUG:root:i=65 residual=9.702271199785173e-06
    DEBUG:root:i=66 residual=8.117508514260408e-06
    DEBUG:root:i=67 residual=6.819955160608515e-06
    DEBUG:root:i=68 residual=5.61670867682551e-06
    DEBUG:root:i=69 residual=4.682388407672988e-06
    DEBUG:root:i=70 residual=3.934549113182584e-06
    DEBUG:root:i=71 residual=3.2853481570782606e-06
    DEBUG:root:i=72 residual=2.6562154289422324e-06
    DEBUG:root:i=73 residual=2.211871333201998e-06
    DEBUG:root:i=74 residual=1.8876633021136513e-06
    DEBUG:root:i=75 residual=1.5732374549770611e-06
    DEBUG:root:i=76 residual=1.3545901538236649e-06
    DEBUG:root:i=77 residual=1.2011980743409367e-06
    DEBUG:root:i=78 residual=1.0143415920538246e-06
    DEBUG:root:i=79 residual=7.319166002162092e-07
    INFO:root:rank=0 pagerank=4.6096e+00 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=2.9870e+00 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 pagerank=2.9672e+00 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 pagerank=2.0175e+00 url=www.lawfareblog.com/senate-examines-threats-homeland
    INFO:root:rank=4 pagerank=1.8771e+00 url=www.lawfareblog.com/what-make-first-day-impeachment-hearings
    INFO:root:rank=5 pagerank=1.8764e+00 url=www.lawfareblog.com/livestream-house-armed-services-committee-hearing-f-35-program
    INFO:root:rank=6 pagerank=1.8695e+00 url=www.lawfareblog.com/whats-house-resolution-impeachment
    INFO:root:rank=7 pagerank=1.7657e+00 url=www.lawfareblog.com/congress-us-policy-toward-syria-and-turkey-overview-recent-hearings
    INFO:root:rank=8 pagerank=1.6809e+00 url=www.lawfareblog.com/summary-david-holmess-deposition-testimony
    INFO:root:rank=9 pagerank=9.8355e-01 url=www.lawfareblog.com/events

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
   DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=6.020042896270752
    DEBUG:root:i=1 residual=4.125221252441406
    DEBUG:root:i=2 residual=3.988121509552002
    DEBUG:root:i=3 residual=3.254075527191162
    DEBUG:root:i=4 residual=2.47937273979187
    DEBUG:root:i=5 residual=1.9349408149719238
    DEBUG:root:i=6 residual=1.6057363748550415
    DEBUG:root:i=7 residual=1.3929798603057861
    DEBUG:root:i=8 residual=1.2307937145233154
    DEBUG:root:i=9 residual=1.0955802202224731
    DEBUG:root:i=10 residual=0.9848984479904175
    DEBUG:root:i=11 residual=0.9026901721954346
    DEBUG:root:i=12 residual=0.8521735668182373
    DEBUG:root:i=13 residual=0.832733154296875
    DEBUG:root:i=14 residual=0.8394593596458435
    DEBUG:root:i=15 residual=0.8648896813392639
    DEBUG:root:i=16 residual=0.9013572931289673
    DEBUG:root:i=17 residual=0.9424902200698853
    DEBUG:root:i=18 residual=0.9836521148681641
    DEBUG:root:i=19 residual=1.0217759609222412
    DEBUG:root:i=20 residual=1.0549975633621216
    DEBUG:root:i=21 residual=1.0823229551315308
    DEBUG:root:i=22 residual=1.1033577919006348
    DEBUG:root:i=23 residual=1.1181122064590454
    DEBUG:root:i=24 residual=1.1268583536148071
    DEBUG:root:i=25 residual=1.1300334930419922
    DEBUG:root:i=26 residual=1.1281652450561523
    DEBUG:root:i=27 residual=1.1218260526657104
    DEBUG:root:i=28 residual=1.1116001605987549
    DEBUG:root:i=29 residual=1.0980511903762817
    DEBUG:root:i=30 residual=1.0817190408706665
    DEBUG:root:i=31 residual=1.0630954504013062
    DEBUG:root:i=32 residual=1.0426439046859741
    DEBUG:root:i=33 residual=1.020767092704773
    DEBUG:root:i=34 residual=0.9978246092796326
    DEBUG:root:i=35 residual=0.9741370677947998
    DEBUG:root:i=36 residual=0.949971079826355
    DEBUG:root:i=37 residual=0.9255669713020325
    DEBUG:root:i=38 residual=0.9011211395263672
    DEBUG:root:i=39 residual=0.8768017292022705
    DEBUG:root:i=40 residual=0.852741539478302
    DEBUG:root:i=41 residual=0.8290578722953796
    DEBUG:root:i=42 residual=0.8058370351791382
    DEBUG:root:i=43 residual=0.7831491231918335
    DEBUG:root:i=44 residual=0.7610563635826111
    DEBUG:root:i=45 residual=0.7395846843719482
    DEBUG:root:i=46 residual=0.7187681794166565
    DEBUG:root:i=47 residual=0.6986279487609863
    DEBUG:root:i=48 residual=0.6791622042655945
    DEBUG:root:i=49 residual=0.6603844165802002
    DEBUG:root:i=50 residual=0.6422829627990723
    DEBUG:root:i=51 residual=0.6248483061790466
    DEBUG:root:i=52 residual=0.6080687642097473
    DEBUG:root:i=53 residual=0.5919331312179565
    DEBUG:root:i=54 residual=0.5764169692993164
    DEBUG:root:i=55 residual=0.5615051984786987
    DEBUG:root:i=56 residual=0.5471781492233276
    DEBUG:root:i=57 residual=0.5334064364433289
    DEBUG:root:i=58 residual=0.5201748609542847
    DEBUG:root:i=59 residual=0.5074607133865356
    DEBUG:root:i=60 residual=0.4952465295791626
    DEBUG:root:i=61 residual=0.48350346088409424
    DEBUG:root:i=62 residual=0.47221437096595764
    DEBUG:root:i=63 residual=0.46135857701301575
    DEBUG:root:i=64 residual=0.4509176015853882
    DEBUG:root:i=65 residual=0.4408646523952484
    DEBUG:root:i=66 residual=0.4311904013156891
    DEBUG:root:i=67 residual=0.4218740463256836
    DEBUG:root:i=68 residual=0.41289955377578735
    DEBUG:root:i=69 residual=0.4042448401451111
    DEBUG:root:i=70 residual=0.3958981931209564
    DEBUG:root:i=71 residual=0.3878469467163086
    DEBUG:root:i=72 residual=0.3800754249095917
    DEBUG:root:i=73 residual=0.3725684881210327
    DEBUG:root:i=74 residual=0.3653225898742676
    DEBUG:root:i=75 residual=0.358310729265213
    DEBUG:root:i=76 residual=0.35152667760849
    DEBUG:root:i=77 residual=0.34497004747390747
    DEBUG:root:i=78 residual=0.3386216461658478
    DEBUG:root:i=79 residual=0.33247560262680054
    DEBUG:root:i=80 residual=0.32650792598724365
    DEBUG:root:i=81 residual=0.3207361102104187
    DEBUG:root:i=82 residual=0.3151337504386902
    DEBUG:root:i=83 residual=0.3096948266029358
    DEBUG:root:i=84 residual=0.3044244050979614
    DEBUG:root:i=85 residual=0.2992852032184601
    DEBUG:root:i=86 residual=0.2943057417869568
    DEBUG:root:i=87 residual=0.28946202993392944
    DEBUG:root:i=88 residual=0.28476420044898987
    DEBUG:root:i=89 residual=0.2801697850227356
    DEBUG:root:i=90 residual=0.27571284770965576
    DEBUG:root:i=91 residual=0.27137741446495056
    DEBUG:root:i=92 residual=0.2671472132205963
    DEBUG:root:i=93 residual=0.2630273699760437
    DEBUG:root:i=94 residual=0.25901228189468384
    DEBUG:root:i=95 residual=0.2550939619541168
    DEBUG:root:i=96 residual=0.2512722909450531
    DEBUG:root:i=97 residual=0.24754422903060913
    DEBUG:root:i=98 residual=0.24390968680381775
    DEBUG:root:i=99 residual=0.24035799503326416
    DEBUG:root:i=100 residual=0.23688633739948273
    DEBUG:root:i=101 residual=0.23349985480308533
    DEBUG:root:i=102 residual=0.23018260300159454
    DEBUG:root:i=103 residual=0.22695006430149078
    DEBUG:root:i=104 residual=0.22378398478031158
    DEBUG:root:i=105 residual=0.2206840068101883
    DEBUG:root:i=106 residual=0.217652827501297
    DEBUG:root:i=107 residual=0.2146795690059662
    DEBUG:root:i=108 residual=0.2117774784564972
    DEBUG:root:i=109 residual=0.20893587172031403
    DEBUG:root:i=110 residual=0.20614410936832428
    DEBUG:root:i=111 residual=0.20341800153255463
    DEBUG:root:i=112 residual=0.20074401795864105
    DEBUG:root:i=113 residual=0.1981198489665985
    DEBUG:root:i=114 residual=0.1955476999282837
    DEBUG:root:i=115 residual=0.1930224448442459
    DEBUG:root:i=116 residual=0.19055438041687012
    DEBUG:root:i=117 residual=0.1881304681301117
    DEBUG:root:i=118 residual=0.18575039505958557
    DEBUG:root:i=119 residual=0.18340912461280823
    DEBUG:root:i=120 residual=0.1811220943927765
    DEBUG:root:i=121 residual=0.17886848747730255
    DEBUG:root:i=122 residual=0.17665328085422516
    DEBUG:root:i=123 residual=0.17448444664478302
    DEBUG:root:i=124 residual=0.1723461002111435
    DEBUG:root:i=125 residual=0.17025388777256012
    DEBUG:root:i=126 residual=0.16819213330745697
    DEBUG:root:i=127 residual=0.16616086661815643
    DEBUG:root:i=128 residual=0.16417038440704346
    DEBUG:root:i=129 residual=0.162215456366539
    DEBUG:root:i=130 residual=0.16028808057308197
    DEBUG:root:i=131 residual=0.1584015190601349
    DEBUG:root:i=132 residual=0.15652671456336975
    DEBUG:root:i=133 residual=0.1547030657529831
    DEBUG:root:i=134 residual=0.15288858115673065
    DEBUG:root:i=135 residual=0.15111719071865082
    DEBUG:root:i=136 residual=0.14937330782413483
    DEBUG:root:i=137 residual=0.14765410125255585
    DEBUG:root:i=138 residual=0.14595192670822144
    DEBUG:root:i=139 residual=0.14429280161857605
    DEBUG:root:i=140 residual=0.14264781773090363
    DEBUG:root:i=141 residual=0.1410275101661682
    DEBUG:root:i=142 residual=0.13943710923194885
    DEBUG:root:i=143 residual=0.13786353170871735
    DEBUG:root:i=144 residual=0.1363118588924408
    DEBUG:root:i=145 residual=0.13479000329971313
    DEBUG:root:i=146 residual=0.1332874894142151
    DEBUG:root:i=147 residual=0.13180691003799438
    DEBUG:root:i=148 residual=0.13034553825855255
    DEBUG:root:i=149 residual=0.12891130149364471
    DEBUG:root:i=150 residual=0.12748321890830994
    DEBUG:root:i=151 residual=0.1260848194360733
    DEBUG:root:i=152 residual=0.12471358478069305
    DEBUG:root:i=153 residual=0.123358815908432
    DEBUG:root:i=154 residual=0.12200222909450531
    DEBUG:root:i=155 residual=0.12068583071231842
    DEBUG:root:i=156 residual=0.11937281489372253
    DEBUG:root:i=157 residual=0.11808683723211288
    DEBUG:root:i=158 residual=0.11682000011205673
    DEBUG:root:i=159 residual=0.11555646359920502
    DEBUG:root:i=160 residual=0.11432253569364548
    DEBUG:root:i=161 residual=0.11309978365898132
    DEBUG:root:i=162 residual=0.11189360171556473
    DEBUG:root:i=163 residual=0.11070119589567184
    DEBUG:root:i=164 residual=0.10952252149581909
    DEBUG:root:i=165 residual=0.1083628460764885
    DEBUG:root:i=166 residual=0.10721965879201889
    DEBUG:root:i=167 residual=0.10609016567468643
    DEBUG:root:i=168 residual=0.10497703403234482
    DEBUG:root:i=169 residual=0.1038646399974823
    DEBUG:root:i=170 residual=0.10277648270130157
    DEBUG:root:i=171 residual=0.10169938951730728
    DEBUG:root:i=172 residual=0.10064125061035156
    DEBUG:root:i=173 residual=0.09959155321121216
    DEBUG:root:i=174 residual=0.09855037182569504
    DEBUG:root:i=175 residual=0.09752538800239563
    DEBUG:root:i=176 residual=0.09651423990726471
    DEBUG:root:i=177 residual=0.09551149606704712
    DEBUG:root:i=178 residual=0.09452761709690094
    DEBUG:root:i=179 residual=0.0935521051287651
    DEBUG:root:i=180 residual=0.09258513897657394
    DEBUG:root:i=181 residual=0.09163179993629456
    DEBUG:root:i=182 residual=0.09069198369979858
    DEBUG:root:i=183 residual=0.08976327627897263
    DEBUG:root:i=184 residual=0.08884037286043167
    DEBUG:root:i=185 residual=0.08793105185031891
    DEBUG:root:i=186 residual=0.08703262358903885
    DEBUG:root:i=187 residual=0.08613483607769012
    DEBUG:root:i=188 residual=0.08526111394166946
    DEBUG:root:i=189 residual=0.08439313620328903
    DEBUG:root:i=190 residual=0.08353336155414581
    DEBUG:root:i=191 residual=0.08268208056688309
    DEBUG:root:i=192 residual=0.08185217529535294
    DEBUG:root:i=193 residual=0.08101750165224075
    DEBUG:root:i=194 residual=0.08020159602165222
    DEBUG:root:i=195 residual=0.0793914720416069
    DEBUG:root:i=196 residual=0.07858698815107346
    DEBUG:root:i=197 residual=0.07779348641633987
    DEBUG:root:i=198 residual=0.07701089978218079
    DEBUG:root:i=199 residual=0.07623130828142166
    DEBUG:root:i=200 residual=0.07546006143093109
    DEBUG:root:i=201 residual=0.07469970732927322
    DEBUG:root:i=202 residual=0.07395809888839722
    DEBUG:root:i=203 residual=0.07321169227361679
    DEBUG:root:i=204 residual=0.07248400896787643
    DEBUG:root:i=205 residual=0.07175420224666595
    DEBUG:root:i=206 residual=0.0710325837135315
    DEBUG:root:i=207 residual=0.0703192874789238
    DEBUG:root:i=208 residual=0.06961939483880997
    DEBUG:root:i=209 residual=0.06891991943120956
    DEBUG:root:i=210 residual=0.06824185699224472
    DEBUG:root:i=211 residual=0.06755886226892471
    DEBUG:root:i=212 residual=0.06688674539327621
    DEBUG:root:i=213 residual=0.06622287631034851
    DEBUG:root:i=214 residual=0.06556463986635208
    DEBUG:root:i=215 residual=0.06489884853363037
    DEBUG:root:i=216 residual=0.06425446271896362
    DEBUG:root:i=217 residual=0.06361830979585648
    DEBUG:root:i=218 residual=0.06298761069774628
    DEBUG:root:i=219 residual=0.06236259266734123
    DEBUG:root:i=220 residual=0.061745792627334595
    DEBUG:root:i=221 residual=0.061142466962337494
    DEBUG:root:i=222 residual=0.06053679808974266
    DEBUG:root:i=223 residual=0.059936728328466415
    DEBUG:root:i=224 residual=0.05933959037065506
    DEBUG:root:i=225 residual=0.05875328555703163
    DEBUG:root:i=226 residual=0.058172557502985
    DEBUG:root:i=227 residual=0.057594750076532364
    DEBUG:root:i=228 residual=0.05703304335474968
    DEBUG:root:i=229 residual=0.05646887049078941
    DEBUG:root:i=230 residual=0.05591043084859848
    DEBUG:root:i=231 residual=0.0553601011633873
    DEBUG:root:i=232 residual=0.054812703281641006
    DEBUG:root:i=233 residual=0.054276082664728165
    DEBUG:root:i=234 residual=0.05373718962073326
    DEBUG:root:i=235 residual=0.05321164429187775
    DEBUG:root:i=236 residual=0.05268106982111931
    DEBUG:root:i=237 residual=0.052166592329740524
    DEBUG:root:i=238 residual=0.05164722725749016
    DEBUG:root:i=239 residual=0.05115426704287529
    DEBUG:root:i=240 residual=0.050648510456085205
    DEBUG:root:i=241 residual=0.05015614628791809
    DEBUG:root:i=242 residual=0.04966137558221817
    DEBUG:root:i=243 residual=0.04917214438319206
    DEBUG:root:i=244 residual=0.04868847504258156
    DEBUG:root:i=245 residual=0.048210348933935165
    DEBUG:root:i=246 residual=0.047737669199705124
    DEBUG:root:i=247 residual=0.04727310687303543
    DEBUG:root:i=248 residual=0.046811431646347046
    DEBUG:root:i=249 residual=0.0463552325963974
    DEBUG:root:i=250 residual=0.045899346470832825
    DEBUG:root:i=251 residual=0.04544892534613609
    DEBUG:root:i=252 residual=0.0450039766728878
    DEBUG:root:i=253 residual=0.044567178934812546
    DEBUG:root:i=254 residual=0.04413323849439621
    DEBUG:root:i=255 residual=0.04370739310979843
    DEBUG:root:i=256 residual=0.04327385500073433
    DEBUG:root:i=257 residual=0.042845916002988815
    DEBUG:root:i=258 residual=0.04243383929133415
    DEBUG:root:i=259 residual=0.042019426822662354
    DEBUG:root:i=260 residual=0.04160787910223007
    DEBUG:root:i=261 residual=0.04119923338294029
    DEBUG:root:i=262 residual=0.040793344378471375
    DEBUG:root:i=263 residual=0.04040338844060898
    DEBUG:root:i=264 residual=0.04000585898756981
    DEBUG:root:i=265 residual=0.03961895406246185
    DEBUG:root:i=266 residual=0.039232343435287476
    DEBUG:root:i=267 residual=0.038851168006658554
    DEBUG:root:i=268 residual=0.03847544640302658
    DEBUG:root:i=269 residual=0.038097310811281204
    DEBUG:root:i=270 residual=0.037727225571870804
    DEBUG:root:i=271 residual=0.03735474869608879
    DEBUG:root:i=272 residual=0.03699294105172157
    DEBUG:root:i=273 residual=0.03663397580385208
    DEBUG:root:i=274 residual=0.03627521172165871
    DEBUG:root:i=275 residual=0.035924479365348816
    DEBUG:root:i=276 residual=0.03557134419679642
    DEBUG:root:i=277 residual=0.035231560468673706
    DEBUG:root:i=278 residual=0.034884076565504074
    DEBUG:root:i=279 residual=0.03455246239900589
    DEBUG:root:i=280 residual=0.03421582281589508
    DEBUG:root:i=281 residual=0.03388207033276558
    DEBUG:root:i=282 residual=0.03355105221271515
    DEBUG:root:i=283 residual=0.0332254096865654
    DEBUG:root:i=284 residual=0.03290000185370445
    DEBUG:root:i=285 residual=0.03258528560400009
    DEBUG:root:i=286 residual=0.03225759044289589
    DEBUG:root:i=287 residual=0.03195630759000778
    DEBUG:root:i=288 residual=0.03163953498005867
    DEBUG:root:i=289 residual=0.03132810816168785
    DEBUG:root:i=290 residual=0.03102993778884411
    DEBUG:root:i=291 residual=0.03072671964764595
    DEBUG:root:i=292 residual=0.030434148386120796
    DEBUG:root:i=293 residual=0.030139079317450523
    DEBUG:root:i=294 residual=0.029846878722310066
    DEBUG:root:i=295 residual=0.029549643397331238
    DEBUG:root:i=296 residual=0.029265573248267174
    DEBUG:root:i=297 residual=0.028986912220716476
    DEBUG:root:i=298 residual=0.02870582416653633
    DEBUG:root:i=299 residual=0.02842230163514614
    DEBUG:root:i=300 residual=0.028152039274573326
    DEBUG:root:i=301 residual=0.02787662297487259
    DEBUG:root:i=302 residual=0.027606643736362457
    DEBUG:root:i=303 residual=0.02733682096004486
    DEBUG:root:i=304 residual=0.027069784700870514
    DEBUG:root:i=305 residual=0.026813248172402382
    DEBUG:root:i=306 residual=0.02654649317264557
    DEBUG:root:i=307 residual=0.026287751272320747
    DEBUG:root:i=308 residual=0.026037011295557022
    DEBUG:root:i=309 residual=0.025788992643356323
    DEBUG:root:i=310 residual=0.025530681014060974
    DEBUG:root:i=311 residual=0.025285597890615463
    DEBUG:root:i=312 residual=0.02504853904247284
    DEBUG:root:i=313 residual=0.024798497557640076
    DEBUG:root:i=314 residual=0.024564305320382118
    DEBUG:root:i=315 residual=0.02431975118815899
    DEBUG:root:i=316 residual=0.024091042578220367
    DEBUG:root:i=317 residual=0.023849429562687874
    DEBUG:root:i=318 residual=0.023631412535905838
    DEBUG:root:i=319 residual=0.023400502279400826
    DEBUG:root:i=320 residual=0.02316451258957386
    DEBUG:root:i=321 residual=0.02294437773525715
    DEBUG:root:i=322 residual=0.022724304348230362
    DEBUG:root:i=323 residual=0.02250179834663868
    DEBUG:root:i=324 residual=0.02228459343314171
    DEBUG:root:i=325 residual=0.02206755429506302
    DEBUG:root:i=326 residual=0.02185588888823986
    DEBUG:root:i=327 residual=0.021641692146658897
    DEBUG:root:i=328 residual=0.021427646279335022
    DEBUG:root:i=329 residual=0.021221598610281944
    DEBUG:root:i=330 residual=0.021020865067839622
    DEBUG:root:i=331 residual=0.020817579701542854
    DEBUG:root:i=332 residual=0.02061453089118004
    DEBUG:root:i=333 residual=0.02041403017938137
    DEBUG:root:i=334 residual=0.020224260166287422
    DEBUG:root:i=335 residual=0.020021509379148483
    DEBUG:root:i=336 residual=0.019831892102956772
    DEBUG:root:i=337 residual=0.01963982544839382
    DEBUG:root:i=338 residual=0.0194452702999115
    DEBUG:root:i=339 residual=0.01925341598689556
    DEBUG:root:i=340 residual=0.019077375531196594
    DEBUG:root:i=341 residual=0.01889095827937126
    DEBUG:root:i=342 residual=0.018707292154431343
    DEBUG:root:i=343 residual=0.01853155344724655
    DEBUG:root:i=344 residual=0.018350718542933464
    DEBUG:root:i=345 residual=0.01817256212234497
    DEBUG:root:i=346 residual=0.01798669993877411
    DEBUG:root:i=347 residual=0.017821837216615677
    DEBUG:root:i=348 residual=0.01764664798974991
    DEBUG:root:i=349 residual=0.01747935265302658
    DEBUG:root:i=350 residual=0.01730436272919178
    DEBUG:root:i=351 residual=0.017139894887804985
    DEBUG:root:i=352 residual=0.016983333975076675
    DEBUG:root:i=353 residual=0.016806066036224365
    DEBUG:root:i=354 residual=0.01665753498673439
    DEBUG:root:i=355 residual=0.016490859910845757
    DEBUG:root:i=356 residual=0.01632952317595482
    DEBUG:root:i=357 residual=0.016165651381015778
    DEBUG:root:i=358 residual=0.01601235754787922
    DEBUG:root:i=359 residual=0.015856511890888214
    DEBUG:root:i=360 residual=0.015703335404396057
    DEBUG:root:i=361 residual=0.01555292122066021
    DEBUG:root:i=362 residual=0.015402555465698242
    DEBUG:root:i=363 residual=0.015254906378686428
    DEBUG:root:i=364 residual=0.015115182846784592
    DEBUG:root:i=365 residual=0.014959907159209251
    DEBUG:root:i=366 residual=0.01481513399630785
    DEBUG:root:i=367 residual=0.014667819254100323
    DEBUG:root:i=368 residual=0.014536287635564804
    DEBUG:root:i=369 residual=0.014394373632967472
    DEBUG:root:i=370 residual=0.014249969273805618
    DEBUG:root:i=371 residual=0.014110835269093513
    DEBUG:root:i=372 residual=0.0139795346185565
    DEBUG:root:i=373 residual=0.013845818117260933
    DEBUG:root:i=374 residual=0.013712156563997269
    DEBUG:root:i=375 residual=0.01358121633529663
    DEBUG:root:i=376 residual=0.013447717763483524
    DEBUG:root:i=377 residual=0.013316886499524117
    DEBUG:root:i=378 residual=0.013188795186579227
    DEBUG:root:i=379 residual=0.01306596864014864
    DEBUG:root:i=380 residual=0.012935363687574863
    DEBUG:root:i=381 residual=0.012817942537367344
    DEBUG:root:i=382 residual=0.012684893794357777
    DEBUG:root:i=383 residual=0.012565013021230698
    DEBUG:root:i=384 residual=0.012442583218216896
    DEBUG:root:i=385 residual=0.012320193462073803
    DEBUG:root:i=386 residual=0.012200471945106983
    DEBUG:root:i=387 residual=0.012086094357073307
    DEBUG:root:i=388 residual=0.011974439024925232
    DEBUG:root:i=389 residual=0.011852310039103031
    DEBUG:root:i=390 residual=0.01173811499029398
    DEBUG:root:i=391 residual=0.011631828732788563
    DEBUG:root:i=392 residual=0.011512583121657372
    DEBUG:root:i=393 residual=0.011406449601054192
    DEBUG:root:i=394 residual=0.011295142583549023
    DEBUG:root:i=395 residual=0.011183938942849636
    DEBUG:root:i=396 residual=0.011075321584939957
    DEBUG:root:i=397 residual=0.01096943486481905
    DEBUG:root:i=398 residual=0.010863649658858776
    DEBUG:root:i=399 residual=0.010744845494627953
    DEBUG:root:i=400 residual=0.010652144439518452
    DEBUG:root:i=401 residual=0.010549105703830719
    DEBUG:root:i=402 residual=0.010446161963045597
    DEBUG:root:i=403 residual=0.01034849788993597
    DEBUG:root:i=404 residual=0.01024565752595663
    DEBUG:root:i=405 residual=0.010145450010895729
    DEBUG:root:i=406 residual=0.01004797127097845
    DEBUG:root:i=407 residual=0.00995052419602871
    DEBUG:root:i=408 residual=0.009853191673755646
    DEBUG:root:i=409 residual=0.009758438915014267
    DEBUG:root:i=410 residual=0.009663799777626991
    DEBUG:root:i=411 residual=0.009579657576978207
    DEBUG:root:i=412 residual=0.009479915723204613
    DEBUG:root:i=413 residual=0.009388066828250885
    DEBUG:root:i=414 residual=0.009298867546021938
    DEBUG:root:i=415 residual=0.00920184887945652
    DEBUG:root:i=416 residual=0.009120570495724678
    DEBUG:root:i=417 residual=0.009028933942317963
    DEBUG:root:i=418 residual=0.008947771042585373
    DEBUG:root:i=419 residual=0.008861428126692772
    DEBUG:root:i=420 residual=0.008772565051913261
    DEBUG:root:i=421 residual=0.008686286397278309
    DEBUG:root:i=422 residual=0.008602735586464405
    DEBUG:root:i=423 residual=0.008516660891473293
    DEBUG:root:i=424 residual=0.008433161303400993
    DEBUG:root:i=425 residual=0.00835499633103609
    DEBUG:root:i=426 residual=0.008274243213236332
    DEBUG:root:i=427 residual=0.008196142502129078
    DEBUG:root:i=428 residual=0.008120683021843433
    DEBUG:root:i=429 residual=0.0080427136272192
    DEBUG:root:i=430 residual=0.007964781485497952
    DEBUG:root:i=431 residual=0.007886870764195919
    DEBUG:root:i=432 residual=0.007808979135006666
    DEBUG:root:i=433 residual=0.0077364384196698666
    DEBUG:root:i=434 residual=0.00765599962323904
    DEBUG:root:i=435 residual=0.007583504542708397
    DEBUG:root:i=436 residual=0.007516256999224424
    DEBUG:root:i=437 residual=0.007441215682774782
    DEBUG:root:i=438 residual=0.007368880789726973
    DEBUG:root:i=439 residual=0.0072964997962117195
    DEBUG:root:i=440 residual=0.007226825226098299
    DEBUG:root:i=441 residual=0.00715725589543581
    DEBUG:root:i=442 residual=0.007085043005645275
    DEBUG:root:i=443 residual=0.007025972008705139
    DEBUG:root:i=444 residual=0.006948571652173996
    DEBUG:root:i=445 residual=0.006886953953653574
    DEBUG:root:i=446 residual=0.00681490171700716
    DEBUG:root:i=447 residual=0.006750781554728746
    DEBUG:root:i=448 residual=0.006691853981465101
    DEBUG:root:i=449 residual=0.006622565910220146
    DEBUG:root:i=450 residual=0.006561145186424255
    DEBUG:root:i=451 residual=0.0064919013530015945
    DEBUG:root:i=452 residual=0.0064279162324965
    DEBUG:root:i=453 residual=0.006371862255036831
    DEBUG:root:i=454 residual=0.006310601718723774
    DEBUG:root:i=455 residual=0.006259768269956112
    DEBUG:root:i=456 residual=0.006182880140841007
    DEBUG:root:i=457 residual=0.006126993801444769
    DEBUG:root:i=458 residual=0.0060632252134382725
    DEBUG:root:i=459 residual=0.006015143357217312
    DEBUG:root:i=460 residual=0.005951491184532642
    DEBUG:root:i=461 residual=0.005893080495297909
    DEBUG:root:i=462 residual=0.005842539947479963
    DEBUG:root:i=463 residual=0.005784179084002972
    DEBUG:root:i=464 residual=0.005725902039557695
    DEBUG:root:i=465 residual=0.005672809202224016
    DEBUG:root:i=466 residual=0.005617198534309864
    DEBUG:root:i=467 residual=0.005566830281168222
    DEBUG:root:i=468 residual=0.00551126291975379
    DEBUG:root:i=469 residual=0.0054504903964698315
    DEBUG:root:i=470 residual=0.005402803421020508
    DEBUG:root:i=471 residual=0.0053499056957662106
    DEBUG:root:i=472 residual=0.005294488742947578
    DEBUG:root:i=473 residual=0.005241704173386097
    DEBUG:root:i=474 residual=0.005188930779695511
    DEBUG:root:i=475 residual=0.0051466431468725204
    DEBUG:root:i=476 residual=0.005091317929327488
    DEBUG:root:i=477 residual=0.005046442616730928
    DEBUG:root:i=478 residual=0.004991207271814346
    DEBUG:root:i=479 residual=0.0049516079016029835
    DEBUG:root:i=480 residual=0.004901648964732885
    DEBUG:root:i=481 residual=0.0048569319769740105
    DEBUG:root:i=482 residual=0.00480961799621582
    DEBUG:root:i=483 residual=0.00476754829287529
    DEBUG:root:i=484 residual=0.004717656876891851
    DEBUG:root:i=485 residual=0.004667853936553001
    DEBUG:root:i=486 residual=0.0046284496784210205
    DEBUG:root:i=487 residual=0.004578675143420696
    DEBUG:root:i=488 residual=0.0045393905602395535
    DEBUG:root:i=489 residual=0.004494883585721254
    DEBUG:root:i=490 residual=0.004439940210431814
    DEBUG:root:i=491 residual=0.004405917599797249
    DEBUG:root:i=492 residual=0.004361468832939863
    DEBUG:root:i=493 residual=0.0043249367736279964
    DEBUG:root:i=494 residual=0.004277843050658703
    DEBUG:root:i=495 residual=0.004241338465362787
    DEBUG:root:i=496 residual=0.004191794898360968
    DEBUG:root:i=497 residual=0.00415265467017889
    DEBUG:root:i=498 residual=0.004113597329705954
    DEBUG:root:i=499 residual=0.0040746210142970085
    DEBUG:root:i=500 residual=0.004040818195790052
    DEBUG:root:i=501 residual=0.004001809749752283
    DEBUG:root:i=502 residual=0.003962875809520483
    DEBUG:root:i=503 residual=0.003921282943338156
    DEBUG:root:i=504 residual=0.0038876112084835768
    DEBUG:root:i=505 residual=0.003840887686237693
    DEBUG:root:i=506 residual=0.003809860674664378
    DEBUG:root:i=507 residual=0.0037710140459239483
    DEBUG:root:i=508 residual=0.003734784899279475
    DEBUG:root:i=509 residual=0.0036985694896429777
    DEBUG:root:i=510 residual=0.003659756388515234
    DEBUG:root:i=511 residual=0.003634099382907152
    DEBUG:root:i=512 residual=0.003590100212022662
    DEBUG:root:i=513 residual=0.003559218253940344
    DEBUG:root:i=514 residual=0.0035257742274552584
    DEBUG:root:i=515 residual=0.0034923190250992775
    DEBUG:root:i=516 residual=0.0034588975831866264
    DEBUG:root:i=517 residual=0.003422857727855444
    DEBUG:root:i=518 residual=0.003389450255781412
    DEBUG:root:i=519 residual=0.0033612940460443497
    DEBUG:root:i=520 residual=0.0033279096242040396
    DEBUG:root:i=521 residual=0.003294616239145398
    DEBUG:root:i=522 residual=0.003256042953580618
    DEBUG:root:i=523 residual=0.003227988025173545
    DEBUG:root:i=524 residual=0.0032025696709752083
    DEBUG:root:i=525 residual=0.0031666390132158995
    DEBUG:root:i=526 residual=0.0031386232003569603
    DEBUG:root:i=527 residual=0.0031080152839422226
    DEBUG:root:i=528 residual=0.003085260046645999
    DEBUG:root:i=529 residual=0.0030416047666221857
    DEBUG:root:i=530 residual=0.003018888644874096
    DEBUG:root:i=531 residual=0.0029857633635401726
    DEBUG:root:i=532 residual=0.002963058650493622
    DEBUG:root:i=533 residual=0.002935202559456229
    DEBUG:root:i=534 residual=0.0029020614456385374
    DEBUG:root:i=535 residual=0.0028742302674800158
    DEBUG:root:i=536 residual=0.002851623110473156
    DEBUG:root:i=537 residual=0.002813296392560005
    DEBUG:root:i=538 residual=0.0027959495782852173
    DEBUG:root:i=539 residual=0.0027681514620780945
    DEBUG:root:i=540 residual=0.0027351484168320894
    DEBUG:root:i=541 residual=0.002715206705033779
    DEBUG:root:i=542 residual=0.0026822309009730816
    DEBUG:root:i=543 residual=0.00265716714784503
    DEBUG:root:i=544 residual=0.0026268111541867256
    DEBUG:root:i=545 residual=0.0026121470145881176
    DEBUG:root:i=546 residual=0.0025897296145558357
    DEBUG:root:i=547 residual=0.002562017645686865
    DEBUG:root:i=548 residual=0.0025343766901642084
    DEBUG:root:i=549 residual=0.002514610067009926
    DEBUG:root:i=550 residual=0.0024896098766475916
    DEBUG:root:i=551 residual=0.0024593304842710495
    DEBUG:root:i=552 residual=0.002436964539811015
    DEBUG:root:i=553 residual=0.002417217707261443
    DEBUG:root:i=554 residual=0.0023922626860439777
    DEBUG:root:i=555 residual=0.0023725293576717377
    DEBUG:root:i=556 residual=0.0023450488224625587
    DEBUG:root:i=557 residual=0.0023175152018666267
    DEBUG:root:i=558 residual=0.0022951955907046795
    DEBUG:root:i=559 residual=0.0022833524271845818
    DEBUG:root:i=560 residual=0.0022637397050857544
    DEBUG:root:i=561 residual=0.0022362391464412212
    DEBUG:root:i=562 residual=0.0022114175371825695
    DEBUG:root:i=563 residual=0.0021892162039875984
    DEBUG:root:i=564 residual=0.0021748063154518604
    DEBUG:root:i=565 residual=0.0021474084351211786
    DEBUG:root:i=566 residual=0.00212783832103014
    DEBUG:root:i=567 residual=0.002105674473568797
    DEBUG:root:i=568 residual=0.0020939717069268227
    DEBUG:root:i=569 residual=0.002066592685878277
    DEBUG:root:i=570 residual=0.0020470849703997374
    DEBUG:root:i=571 residual=0.002027547685429454
    DEBUG:root:i=572 residual=0.0020158991683274508
    DEBUG:root:i=573 residual=0.0019937660545110703
    DEBUG:root:i=574 residual=0.0019690515473484993
    DEBUG:root:i=575 residual=0.001954851672053337
    DEBUG:root:i=576 residual=0.001935369218699634
    DEBUG:root:i=577 residual=0.0019132939632982016
    DEBUG:root:i=578 residual=0.0018991149263456464
    DEBUG:root:i=579 residual=0.0018796671647578478
    DEBUG:root:i=580 residual=0.0018602757481858134
    DEBUG:root:i=581 residual=0.0018435134552419186
    DEBUG:root:i=582 residual=0.0018240765202790499
    DEBUG:root:i=583 residual=0.0018021204741671681
    DEBUG:root:i=584 residual=0.001787981833331287
    DEBUG:root:i=585 residual=0.0017712466651573777
    DEBUG:root:i=586 residual=0.0017492915503680706
    DEBUG:root:i=587 residual=0.0017404077807441354
    DEBUG:root:i=588 residual=0.0017184842145070434
    DEBUG:root:i=589 residual=0.0017070004250854254
    DEBUG:root:i=590 residual=0.001685072435066104
    DEBUG:root:i=591 residual=0.0016709900228306651
    DEBUG:root:i=592 residual=0.0016569198342040181
    DEBUG:root:i=593 residual=0.0016323986928910017
    DEBUG:root:i=594 residual=0.0016262447461485863
    DEBUG:root:i=595 residual=0.001609580242075026
    DEBUG:root:i=596 residual=0.001592915621586144
    DEBUG:root:i=597 residual=0.0015789384488016367
    DEBUG:root:i=598 residual=0.0015596722951158881
    DEBUG:root:i=599 residual=0.0015404826262965798
    DEBUG:root:i=600 residual=0.0015316919889301062
    DEBUG:root:i=601 residual=0.001517742988653481
    DEBUG:root:i=602 residual=0.0015011237701401114
    DEBUG:root:i=603 residual=0.0014898012159392238
    DEBUG:root:i=604 residual=0.0014706386718899012
    DEBUG:root:i=605 residual=0.001456642639823258
    DEBUG:root:i=606 residual=0.0014453367330133915
    DEBUG:root:i=607 residual=0.0014314199797809124
    DEBUG:root:i=608 residual=0.001420116750523448
    DEBUG:root:i=609 residual=0.00140882667619735
    DEBUG:root:i=610 residual=0.0013870951952412724
    DEBUG:root:i=611 residual=0.0013810390373691916
    DEBUG:root:i=612 residual=0.0013593105832114816
    DEBUG:root:i=613 residual=0.0013532746816053987
    DEBUG:root:i=614 residual=0.0013419988099485636
    DEBUG:root:i=615 residual=0.001330751576460898
    DEBUG:root:i=616 residual=0.0013090413995087147
    DEBUG:root:i=617 residual=0.0012978584272786975
    DEBUG:root:i=618 residual=0.0012866270262748003
    DEBUG:root:i=619 residual=0.00127015495672822
    DEBUG:root:i=620 residual=0.0012641455978155136
    DEBUG:root:i=621 residual=0.0012477583950385451
    DEBUG:root:i=622 residual=0.0012417638208717108
    DEBUG:root:i=623 residual=0.001230613561347127
    DEBUG:root:i=624 residual=0.0012141752522438765
    DEBUG:root:i=625 residual=0.0011978024849668145
    DEBUG:root:i=626 residual=0.0011918306117877364
    DEBUG:root:i=627 residual=0.0011806900147348642
    DEBUG:root:i=628 residual=0.0011747267562896013
    DEBUG:root:i=629 residual=0.0011636015260592103
    DEBUG:root:i=630 residual=0.0011472511105239391
    DEBUG:root:i=631 residual=0.0011308518005535007
    DEBUG:root:i=632 residual=0.0011249672388657928
    DEBUG:root:i=633 residual=0.0011138564441353083
    DEBUG:root:i=634 residual=0.0011027560103684664
    DEBUG:root:i=635 residual=0.001091589918360114
    DEBUG:root:i=636 residual=0.001085722935386002
    DEBUG:root:i=637 residual=0.0010745679028332233
    DEBUG:root:i=638 residual=0.001063489937223494
    DEBUG:root:i=639 residual=0.001047176425345242
    DEBUG:root:i=640 residual=0.0010361041640862823
    DEBUG:root:i=641 residual=0.0010250348132103682
    DEBUG:root:i=642 residual=0.0010244199074804783
    DEBUG:root:i=643 residual=0.00100551953073591
    DEBUG:root:i=644 residual=0.0009996809531003237
    DEBUG:root:i=645 residual=0.0009886316256597638
    DEBUG:root:i=646 residual=0.0009801913984119892
    DEBUG:root:i=647 residual=0.0009692034218460321
    DEBUG:root:i=648 residual=0.0009633867302909493
    DEBUG:root:i=649 residual=0.0009445146424695849
    DEBUG:root:i=650 residual=0.0009439245914109051
    DEBUG:root:i=651 residual=0.0009329643798992038
    DEBUG:root:i=652 residual=0.0009219379862770438
    DEBUG:root:i=653 residual=0.0009161341586150229
    DEBUG:root:i=654 residual=0.0009104025666601956
    DEBUG:root:i=655 residual=0.0008993893279694021
    DEBUG:root:i=656 residual=0.0008883815607987344
    DEBUG:root:i=657 residual=0.0008878817316144705
    DEBUG:root:i=658 residual=0.0008664243505336344
    DEBUG:root:i=659 residual=0.000863317574840039
    DEBUG:root:i=660 residual=0.0008549378253519535
    DEBUG:root:i=661 residual=0.0008544537704437971
    DEBUG:root:i=662 residual=0.0008382318774238229
    DEBUG:root:i=663 residual=0.0008325358503498137
    DEBUG:root:i=664 residual=0.000829444034025073
    DEBUG:root:i=665 residual=0.000813236110843718
    DEBUG:root:i=666 residual=0.0008101633284240961
    DEBUG:root:i=667 residual=0.0007991926395334303
    DEBUG:root:i=668 residual=0.0007882787613198161
    DEBUG:root:i=669 residual=0.0007826032815501094
    DEBUG:root:i=670 residual=0.0007717027910985053
    DEBUG:root:i=671 residual=0.000771176943089813
    DEBUG:root:i=672 residual=0.0007576714269816875
    DEBUG:root:i=673 residual=0.000754625943955034
    DEBUG:root:i=674 residual=0.0007489618146792054
    DEBUG:root:i=675 residual=0.0007406887016259134
    DEBUG:root:i=676 residual=0.0007376429857686162
    DEBUG:root:i=677 residual=0.0007319248979911208
    DEBUG:root:i=678 residual=0.0007210491457954049
    DEBUG:root:i=679 residual=0.0007206163718365133
    DEBUG:root:i=680 residual=0.0006992930429987609
    DEBUG:root:i=681 residual=0.0006988705135881901
    DEBUG:root:i=682 residual=0.0006932406104169786
    DEBUG:root:i=683 residual=0.0006849775090813637
    DEBUG:root:i=684 residual=0.0006767375161871314
    DEBUG:root:i=685 residual=0.0006737090880051255
    DEBUG:root:i=686 residual=0.0006654706667177379
    DEBUG:root:i=687 residual=0.0006598408217541873
    DEBUG:root:i=688 residual=0.0006568183889612556
    DEBUG:root:i=689 residual=0.0006460460717789829
    DEBUG:root:i=690 residual=0.0006430446519516408
    DEBUG:root:i=691 residual=0.0006374275544658303
    DEBUG:root:i=692 residual=0.000629190995823592
    DEBUG:root:i=693 residual=0.0006209648563526571
    DEBUG:root:i=694 residual=0.0006127431406639516
    DEBUG:root:i=695 residual=0.0006071989773772657
    DEBUG:root:i=696 residual=0.0006042146123945713
    DEBUG:root:i=697 residual=0.0006012204685248435
    DEBUG:root:i=698 residual=0.0005930150509811938
    DEBUG:root:i=699 residual=0.0005848560831509531
    DEBUG:root:i=700 residual=0.0005818845820613205
    DEBUG:root:i=701 residual=0.0005815132753923535
    DEBUG:root:i=702 residual=0.0005655352142639458
    DEBUG:root:i=703 residual=0.0005651727551594377
    DEBUG:root:i=704 residual=0.0005622152821160853
    DEBUG:root:i=705 residual=0.0005593057721853256
    DEBUG:root:i=706 residual=0.0005484860157594085
    DEBUG:root:i=707 residual=0.000542967754881829
    DEBUG:root:i=708 residual=0.0005347802070900798
    DEBUG:root:i=709 residual=0.000529263517819345
    DEBUG:root:i=710 residual=0.0005236873985268176
    DEBUG:root:i=711 residual=0.0005233478150330484
    DEBUG:root:i=712 residual=0.0005178435822017491
    DEBUG:root:i=713 residual=0.0005070458864793181
    DEBUG:root:i=714 residual=0.0005067781312391162
    DEBUG:root:i=715 residual=0.0005012766923755407
    DEBUG:root:i=716 residual=0.0005009389715269208
    DEBUG:root:i=717 residual=0.0004928400157950819
    DEBUG:root:i=718 residual=0.000487281649839133
    DEBUG:root:i=719 residual=0.0004870274569839239
    DEBUG:root:i=720 residual=0.00048146359040401876
    DEBUG:root:i=721 residual=0.00047335680574178696
    DEBUG:root:i=722 residual=0.0004705066967289895
    DEBUG:root:i=723 residual=0.0004701678117271513
    DEBUG:root:i=724 residual=0.00046469923108816147
    DEBUG:root:i=725 residual=0.00045398849761113524
    DEBUG:root:i=726 residual=0.0004484480887185782
    DEBUG:root:i=727 residual=0.00044558208901435137
    DEBUG:root:i=728 residual=0.0004453264700714499
    DEBUG:root:i=729 residual=0.000439794355770573
    DEBUG:root:i=730 residual=0.00042909360490739346
    DEBUG:root:i=731 residual=0.00042625205242075026
    DEBUG:root:i=732 residual=0.0004260087735019624
    DEBUG:root:i=733 residual=0.0004231568018440157
    DEBUG:root:i=734 residual=0.00041762093314900994
    DEBUG:root:i=735 residual=0.00041478368802927434
    DEBUG:root:i=736 residual=0.0004119252262171358
    DEBUG:root:i=737 residual=0.00040647145942784846
    DEBUG:root:i=738 residual=0.0004036272584926337
    DEBUG:root:i=739 residual=0.00039810844464227557
    DEBUG:root:i=740 residual=0.0003978848981205374
    DEBUG:root:i=741 residual=0.0003898100694641471
    DEBUG:root:i=742 residual=0.00038958017830736935
    DEBUG:root:i=743 residual=0.0003893687389791012
    DEBUG:root:i=744 residual=0.00037608316051773727
    DEBUG:root:i=745 residual=0.0003758645325433463
    DEBUG:root:i=746 residual=0.00037564680678769946
    DEBUG:root:i=747 residual=0.00037281474214978516
    DEBUG:root:i=748 residual=0.000364694424206391
    DEBUG:root:i=749 residual=0.00035926190321333706
    DEBUG:root:i=750 residual=0.0003590487758629024
    DEBUG:root:i=751 residual=0.0003562089113984257
    DEBUG:root:i=752 residual=0.00034817398409359157
    DEBUG:root:i=753 residual=0.0003479659208096564
    DEBUG:root:i=754 residual=0.0003425286849960685
    DEBUG:root:i=755 residual=0.0003370912454556674
    DEBUG:root:i=756 residual=0.00033689234987832606
    DEBUG:root:i=757 residual=0.0003366861492395401
    DEBUG:root:i=758 residual=0.00033386764698661864
    DEBUG:root:i=759 residual=0.0003336668887641281
    DEBUG:root:i=760 residual=0.00032823829678818583
    DEBUG:root:i=761 residual=0.0003176524769514799
    DEBUG:root:i=762 residual=0.00031750218477100134
    DEBUG:root:i=763 residual=0.0003173061413690448
    DEBUG:root:i=764 residual=0.00031446764478459954
    DEBUG:root:i=765 residual=0.0003116506268270314
    DEBUG:root:i=766 residual=0.0003114635474048555
    DEBUG:root:i=767 residual=0.00030603029881604016
    DEBUG:root:i=768 residual=0.00030585567583329976
    DEBUG:root:i=769 residual=0.0003004397440236062
    DEBUG:root:i=770 residual=0.00029508545412681997
    DEBUG:root:i=771 residual=0.0002949042827822268
    DEBUG:root:i=772 residual=0.0002947278553619981
    DEBUG:root:i=773 residual=0.00029194235685281456
    DEBUG:root:i=774 residual=0.0002865311107598245
    DEBUG:root:i=775 residual=0.0002811263839248568
    DEBUG:root:i=776 residual=0.0002810146543197334
    DEBUG:root:i=777 residual=0.00027561295428313315
    DEBUG:root:i=778 residual=0.00027285743271932006
    DEBUG:root:i=779 residual=0.0002726754464674741
    DEBUG:root:i=780 residual=0.00026727659860625863
    DEBUG:root:i=781 residual=0.00025929807452484965
    DEBUG:root:i=782 residual=0.0002591207448858768
    DEBUG:root:i=783 residual=0.00025894594728015363
    DEBUG:root:i=784 residual=0.00025878180167637765
    DEBUG:root:i=785 residual=0.00025603926042094827
    DEBUG:root:i=786 residual=0.00025326889590360224
    DEBUG:root:i=787 residual=0.0002531111822463572
    DEBUG:root:i=788 residual=0.000252940459176898
    DEBUG:root:i=789 residual=0.00024761739769019186
    DEBUG:root:i=790 residual=0.0002474508073646575
    DEBUG:root:i=791 residual=0.0002394221373833716
    DEBUG:root:i=792 residual=0.00023933235206641257
    DEBUG:root:i=793 residual=0.00023917289217934012
    DEBUG:root:i=794 residual=0.0002364124811720103
    DEBUG:root:i=795 residual=0.00023109388712327927
    DEBUG:root:i=796 residual=0.00023093458730727434
    DEBUG:root:i=797 residual=0.00022554518363904208
    DEBUG:root:i=798 residual=0.00022545251704286784
    DEBUG:root:i=799 residual=0.00022529842681251466
    DEBUG:root:i=800 residual=0.00022514554439112544
    DEBUG:root:i=801 residual=0.00022242829436436296
    DEBUG:root:i=802 residual=0.00022227839508559555
    DEBUG:root:i=803 residual=0.00021689046116080135
    DEBUG:root:i=804 residual=0.00021687717526219785
    DEBUG:root:i=805 residual=0.0002167235070373863
    DEBUG:root:i=806 residual=0.0002061167760984972
    DEBUG:root:i=807 residual=0.00020603727898560464
    DEBUG:root:i=808 residual=0.00020588547340594232
    DEBUG:root:i=809 residual=0.00020579893316607922
    DEBUG:root:i=810 residual=0.00020566099556162953
    DEBUG:root:i=811 residual=0.00020558643154799938
    DEBUG:root:i=812 residual=0.00019753929518628865
    DEBUG:root:i=813 residual=0.00019738648552447557
    DEBUG:root:i=814 residual=0.00019475686713121831
    DEBUG:root:i=815 residual=0.00019461981719359756
    DEBUG:root:i=816 residual=0.00018931730301119387
    DEBUG:root:i=817 residual=0.00018386928422842175
    DEBUG:root:i=818 residual=0.00017856963677331805
    DEBUG:root:i=819 residual=0.00017842352099251002
    DEBUG:root:i=820 residual=0.00017835661128629
    DEBUG:root:i=821 residual=0.00017820952052716166
    DEBUG:root:i=822 residual=0.00017813344311434776
    DEBUG:root:i=823 residual=0.00017800556088332087
    DEBUG:root:i=824 residual=0.0001779327285476029
    DEBUG:root:i=825 residual=0.0001752006501192227
    DEBUG:root:i=826 residual=0.0001751360105117783
    DEBUG:root:i=827 residual=0.00016976523329503834
    DEBUG:root:i=828 residual=0.00016970204887911677
    DEBUG:root:i=829 residual=0.0001695714017841965
    DEBUG:root:i=830 residual=0.00016950690769590437
    DEBUG:root:i=831 residual=0.00016673341451678425
    DEBUG:root:i=832 residual=0.0001640820555621758
    DEBUG:root:i=833 residual=0.00016395268903579563
    DEBUG:root:i=834 residual=0.00015865500608924776
    DEBUG:root:i=835 residual=0.00015852988872211426
    DEBUG:root:i=836 residual=0.00015582272317260504
    DEBUG:root:i=837 residual=0.0001556930219521746
    DEBUG:root:i=838 residual=0.0001556356146465987
    DEBUG:root:i=839 residual=0.00014776054013054818
    DEBUG:root:i=840 residual=0.00014763201761525124
    DEBUG:root:i=841 residual=0.0001449388946639374
    DEBUG:root:i=842 residual=0.0001448052207706496
    DEBUG:root:i=843 residual=0.00014474680938292295
    DEBUG:root:i=844 residual=0.00014462323451880366
    DEBUG:root:i=845 residual=0.00014456971257459372
    DEBUG:root:i=846 residual=0.00014451293100137264
    DEBUG:root:i=847 residual=0.00014179968275129795
    DEBUG:root:i=848 residual=0.00013910052075516433
    DEBUG:root:i=849 residual=0.00014163018204271793
    DEBUG:root:i=850 residual=0.00014157760597299784
    DEBUG:root:i=851 residual=0.0001388825330650434
    DEBUG:root:i=852 residual=0.00014141679275780916
    DEBUG:root:i=853 residual=0.00013088092964608222
    DEBUG:root:i=854 residual=0.00012812270142603666
    DEBUG:root:i=855 residual=0.00013071607099846005
    DEBUG:root:i=856 residual=0.00012543794582597911
    DEBUG:root:i=857 residual=0.00012790616892743856
    DEBUG:root:i=858 residual=0.00012527099170256406
    DEBUG:root:i=859 residual=0.00012521441385615617
    DEBUG:root:i=860 residual=0.00012769341992679983
    DEBUG:root:i=861 residual=0.00012505840277299285
    DEBUG:root:i=862 residual=0.00012500972661655396
    DEBUG:root:i=863 residual=0.00012749519373755902
    DEBUG:root:i=864 residual=0.00012485742627177387
    DEBUG:root:i=865 residual=0.00012209740816615522
    DEBUG:root:i=866 residual=0.00012204954691696912
    DEBUG:root:i=867 residual=0.00012200910714454949
    DEBUG:root:i=868 residual=0.00012190121924504638
    DEBUG:root:i=869 residual=0.0001140262684202753
    DEBUG:root:i=870 residual=0.00010875920270336792
    DEBUG:root:i=871 residual=0.00010871427366510034
    DEBUG:root:i=872 residual=0.00010596108040772378
    DEBUG:root:i=873 residual=0.00010855405707843602
    DEBUG:root:i=874 residual=0.00010851038678083569
    DEBUG:root:i=875 residual=0.00010575445048743859
    DEBUG:root:i=876 residual=9.79233009275049e-05
    DEBUG:root:i=877 residual=9.52434420469217e-05
    DEBUG:root:i=878 residual=9.256997873308137e-05
    DEBUG:root:i=879 residual=9.508711082162336e-05
    DEBUG:root:i=880 residual=9.248323476640508e-05
    DEBUG:root:i=881 residual=9.235959441866726e-05
    DEBUG:root:i=882 residual=9.488635987509042e-05
    DEBUG:root:i=883 residual=9.22785620787181e-05
    DEBUG:root:i=884 residual=9.2239111836534e-05
    DEBUG:root:i=885 residual=9.470272198086604e-05
    DEBUG:root:i=886 residual=9.209451673086733e-05
    DEBUG:root:i=887 residual=8.940506813814864e-05
    DEBUG:root:i=888 residual=8.409285510424525e-05
    DEBUG:root:i=889 residual=8.149530185619369e-05
    DEBUG:root:i=890 residual=8.400871593039483e-05
    DEBUG:root:i=891 residual=8.141623402480036e-05
    DEBUG:root:i=892 residual=8.13033984741196e-05
    DEBUG:root:i=893 residual=8.38207415654324e-05
    DEBUG:root:i=894 residual=8.122893632389605e-05
    DEBUG:root:i=895 residual=8.118931873468682e-05
    DEBUG:root:i=896 residual=8.364540553884581e-05
    DEBUG:root:i=897 residual=8.104927110252902e-05
    DEBUG:root:i=898 residual=8.100720151560381e-05
    DEBUG:root:i=899 residual=8.353872544830665e-05
    DEBUG:root:i=900 residual=8.087014430202544e-05
    DEBUG:root:i=901 residual=7.81801572884433e-05
    DEBUG:root:i=902 residual=7.814905256964266e-05
    DEBUG:root:i=903 residual=7.811649265931919e-05
    DEBUG:root:i=904 residual=7.801598258083686e-05
    DEBUG:root:i=905 residual=7.798431761329994e-05
    DEBUG:root:i=906 residual=7.795375131536275e-05
    DEBUG:root:i=907 residual=7.792600081302226e-05
    DEBUG:root:i=908 residual=7.788705261191353e-05
    DEBUG:root:i=909 residual=7.256874232552946e-05
    DEBUG:root:i=910 residual=7.519492646679282e-05
    DEBUG:root:i=911 residual=6.73010217724368e-05
    DEBUG:root:i=912 residual=6.726975698256865e-05
    DEBUG:root:i=913 residual=6.716427742503583e-05
    DEBUG:root:i=914 residual=6.459476571762934e-05
    DEBUG:root:i=915 residual=6.456505798269063e-05
    DEBUG:root:i=916 residual=5.9363770560594276e-05
    DEBUG:root:i=917 residual=5.933206557529047e-05
    DEBUG:root:i=918 residual=6.174813461257145e-05
    DEBUG:root:i=919 residual=5.918132228543982e-05
    DEBUG:root:i=920 residual=5.915137444389984e-05
    DEBUG:root:i=921 residual=6.165008380776271e-05
    DEBUG:root:i=922 residual=5.9090300055686384e-05
    DEBUG:root:i=923 residual=5.898674498894252e-05
    DEBUG:root:i=924 residual=6.149553519207984e-05
    DEBUG:root:i=925 residual=5.893187699257396e-05
    DEBUG:root:i=926 residual=5.8904166508000344e-05
    DEBUG:root:i=927 residual=6.141603080322966e-05
    DEBUG:root:i=928 residual=5.877743024029769e-05
    DEBUG:root:i=929 residual=5.875180795555934e-05
    DEBUG:root:i=930 residual=6.127458618720993e-05
    DEBUG:root:i=931 residual=5.869523738510907e-05
    DEBUG:root:i=932 residual=5.8669760619523004e-05
    DEBUG:root:i=933 residual=6.119559111539274e-05
    DEBUG:root:i=934 residual=5.855519702890888e-05
    DEBUG:root:i=935 residual=5.586108090938069e-05
    DEBUG:root:i=936 residual=5.5834541853982955e-05
    DEBUG:root:i=937 residual=5.5811364291002974e-05
    DEBUG:root:i=938 residual=5.5790646001696587e-05
    DEBUG:root:i=939 residual=5.5769382015569136e-05
    DEBUG:root:i=940 residual=5.567691187025048e-05
    DEBUG:root:i=941 residual=5.044352656113915e-05
    DEBUG:root:i=942 residual=5.309315019985661e-05
    DEBUG:root:i=943 residual=5.039918687543832e-05
    DEBUG:root:i=944 residual=5.304797014105134e-05
    DEBUG:root:i=945 residual=5.3019753977423534e-05
    DEBUG:root:i=946 residual=5.025593418395147e-05
    DEBUG:root:i=947 residual=5.291389243211597e-05
    DEBUG:root:i=948 residual=5.289239561534487e-05
    DEBUG:root:i=949 residual=5.018964293412864e-05
    DEBUG:root:i=950 residual=5.2856423280900344e-05
    DEBUG:root:i=951 residual=5.283741484163329e-05
    DEBUG:root:i=952 residual=5.013239206164144e-05
    DEBUG:root:i=953 residual=5.0048023695126176e-05
    DEBUG:root:i=954 residual=4.2307750845793635e-05
    DEBUG:root:i=955 residual=4.228599937050603e-05
    DEBUG:root:i=956 residual=3.959792593377642e-05
    DEBUG:root:i=957 residual=4.2242802010150626e-05
    DEBUG:root:i=958 residual=4.222356074023992e-05
    DEBUG:root:i=959 residual=3.953268242185004e-05
    DEBUG:root:i=960 residual=4.2102063162019476e-05
    DEBUG:root:i=961 residual=4.208508107694797e-05
    DEBUG:root:i=962 residual=3.93900518247392e-05
    DEBUG:root:i=963 residual=4.204825745546259e-05
    DEBUG:root:i=964 residual=4.203252319712192e-05
    DEBUG:root:i=965 residual=3.933315383619629e-05
    DEBUG:root:i=966 residual=4.1995499486802146e-05
    DEBUG:root:i=967 residual=4.191562038613483e-05
    DEBUG:root:i=968 residual=3.92032197851222e-05
    DEBUG:root:i=969 residual=4.187463491689414e-05
    DEBUG:root:i=970 residual=4.186089427093975e-05
    DEBUG:root:i=971 residual=3.915277557098307e-05
    DEBUG:root:i=972 residual=4.183037526672706e-05
    DEBUG:root:i=973 residual=4.181467375019565e-05
    DEBUG:root:i=974 residual=3.910185841959901e-05
    DEBUG:root:i=975 residual=4.178543167654425e-05
    DEBUG:root:i=976 residual=4.170839019934647e-05
    DEBUG:root:i=977 residual=3.89879678550642e-05
    DEBUG:root:i=978 residual=4.168641316937283e-05
    DEBUG:root:i=979 residual=4.1661533032311127e-05
    DEBUG:root:i=980 residual=3.894009569194168e-05
    DEBUG:root:i=981 residual=4.163842095294967e-05
    DEBUG:root:i=982 residual=3.64003935828805e-05
    DEBUG:root:i=983 residual=3.8904861867194995e-05
    DEBUG:root:i=984 residual=3.1128398404689506e-05
    DEBUG:root:i=985 residual=3.3590335078770295e-05
    DEBUG:root:i=986 residual=3.3571632229723036e-05
    DEBUG:root:i=987 residual=3.3563388569746166e-05
    DEBUG:root:i=988 residual=3.354204091010615e-05
    DEBUG:root:i=989 residual=3.3534019166836515e-05
    DEBUG:root:i=990 residual=3.351558916619979e-05
    DEBUG:root:i=991 residual=3.350643237354234e-05
    DEBUG:root:i=992 residual=3.3488387998659164e-05
    DEBUG:root:i=993 residual=3.347777601447888e-05
    DEBUG:root:i=994 residual=3.340568582643755e-05
    DEBUG:root:i=995 residual=3.089478559559211e-05
    DEBUG:root:i=996 residual=2.818497523549013e-05
    DEBUG:root:i=997 residual=3.087952427449636e-05
    DEBUG:root:i=998 residual=3.086068682023324e-05
    DEBUG:root:i=999 residual=2.8144842872279696e-05
    INFO:root:rank=0 pagerank=5.2386e+01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=5.2386e+01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=7.9439e+00 url=www.lawfareblog.com/cost-using-zero-days
    INFO:root:rank=3 pagerank=2.3700e+00 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
    INFO:root:rank=4 pagerank=1.5530e+00 url=www.lawfareblog.com/events
    INFO:root:rank=5 pagerank=1.1867e+00 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
    INFO:root:rank=6 pagerank=1.1867e+00 url=www.lawfareblog.com/water-wars-drill-maybe-drill
    INFO:root:rank=7 pagerank=1.1867e+00 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
    INFO:root:rank=8 pagerank=1.1867e+00 url=www.lawfareblog.com/water-wars-us-china-divide-shangri-la
    INFO:root:rank=9 pagerank=1.1867e+00 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
   
   ```

   Task 2, part 1:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
   INFO:root:rank=0 pagerank=8.8870e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=8.8867e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.8256e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 pagerank=1.4907e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
    INFO:root:rank=4 pagerank=1.4907e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
    INFO:root:rank=5 pagerank=1.0729e-01 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=6 pagerank=1.0199e-01 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
    INFO:root:rank=7 pagerank=1.0199e-01 url=www.lawfareblog.com/britains-coronavirus-response
    INFO:root:rank=8 pagerank=9.4298e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
    INFO:root:rank=9 pagerank=8.7207e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
   ```

   Task 2, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
   INFO:root:rank=0 pagerank=8.8870e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=8.8867e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.8256e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 pagerank=1.0729e-01 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=4 pagerank=9.4298e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
    INFO:root:rank=5 pagerank=7.9633e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
    INFO:root:rank=6 pagerank=7.5307e-02 url=www.lawfareblog.com/limits-world-health-organization
    INFO:root:rank=7 pagerank=6.8115e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
    INFO:root:rank=8 pagerank=6.4847e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
    INFO:root:rank=9 pagerank=6.4847e-02 url=www.lawfareblog.com/livestream-house-armed-services-committee-holds-hearing-priorities-missile-defense
   ```

**Update using Pagerank2.py**

    python3 pagerank2.py --data=data/lawfareblog.csv.gz --search_query='trump' --s_weight=.01
    [==================================================] 100.0% 128.1/128.1MB downloaded
    INFO:gensim.models.keyedvectors:precomputing L2-norms of word weight vectors
    INFO:root:rank=0 pagerank=0.015782557427883148 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=0.015233838930726051 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
    INFO:root:rank=2 pagerank=0.01512967050075531 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
    INFO:root:rank=3 pagerank=0.014659898355603218 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
    INFO:root:rank=4 pagerank=0.014593398198485374 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
    INFO:root:rank=5 pagerank=0.014307133853435516 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
    INFO:root:rank=6 pagerank=0.014093476347625256 url=www.lawfareblog.com/why-trump-cant-buy-greenland
    INFO:root:rank=7 pagerank=0.013759083114564419 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
    INFO:root:rank=8 pagerank=0.013450872153043747 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
    INFO:root:rank=9 pagerank=0.013448437675833702 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors




1. Ensure that all your changes to the `pagerank.py` and `README.md` files are committed to your repo and pushed to github.

1. Get at least 5 stars on your repo.
   (You may trade stars with other students in the class.)

   > **NOTE:**
   > 
   > Recruiters use github profiles to determine who to hire,
   > and pagerank is used to rank user profiles and projects.
   > Links in this graph correspond to who has starred/followed who's repo.
   > By getting more stars on your repo, you'll be increasing your github pagerank, which increases the likelihood that recruiters will hire you.
   > To see an example, [perform a search for `data mining`](https://github.com/search?q=data+mining).
   > Notice that the results are returned "approximately" ranked by the number of stars,
   > but because "some stars count more than others" the results are not exactly ranked by the number of stars.
   > (I asked you not to fork this repo because forks are ranked lower than non-forks.)
   >
   > In some sense, we are doing a "dual problem" to data mining by getting these stars.
   > Recruiters are using data mining to find out who the best people to recruit are,
   > and we are hacking their data mining algorithms by making those algorithms select you instead of someone else.
   >
   > If you're interested in exploring this idea further, here's a python tutorial for extracting GitHub's social graph: <https://www.oreilly.com/library/view/mining-the-social/9781449368180/ch07.html> ; if you're interested in learning more about how recruiters use github profiles, read this Hacker News post: <https://news.ycombinator.com/item?id=19413348>.

1. Submit the url of your repo to sakai.

   Each part is worth 2 points, for 12 points overall.
