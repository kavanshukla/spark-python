from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql import Row
from pyspark.sql.functions import lit
from pyspark.sql import SQLContext
import sys

conf = SparkConf().setAppName('Movie Recommendation')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


movie_data = sys.argv[1]
user_data = sys.argv[2]
output = sys.argv[3]


userId = 112132212
movie = Row("id","movieName")

movie_table = sc.textFile(movie_data+str("/movies.dat"))
rating_table = sc.textFile(movie_data+str("/ratings.dat"))
user_data_table = sc.textFile(movie_data+str("/users.dat"))
new_user = sc.textFile(user_data)

movieRDD = movie_table.map(lambda movie:movie.split("::"))

ratingDF = (rating_table.map(lambda rating:rating.split("::"))
                         .map(lambda rate:(int(rate[0]),int(rate[1]),float(rate[2])))
                         .map(lambda (uid,mid,rate):Rating(uid,mid,rate))).toDF()


newUserRDD = new_user.map(lambda movie:movie.split(" ",1))

joinRDD = movieRDD.cartesian(newUserRDD)
joinRDD = (joinRDD.map(lambda (movie,umovie):(movie[0],movie[1],
                                            umovie[0],umovie[1]))
                      .map(lambda (id,movie,urate,umovie):(umovie,
                                            (id,urate,levenshtein(movie,umovie))))
                      .reduceByKey(lambda x1, x2: min(x1, x2, key=lambda x: x[-1])))

userMovie = (joinRDD.map(lambda (key,value):(userId,value[0],value[1]))
                      .map(lambda (uid,mid,rate):Rating(uid,int(mid),float(rate)))
                      ).toDF()

trainDF = ratingDF.cache()

myratedMovieTrain = trainDF.unionAll(userMovie)

model = ALS.train(myratedMovieTrain, 10, 10,lambda_=0.01)

userRatedMovie = userMovie.rdd.map(lambda (uid,mid,rate):mid).collect()

movieDF = movieRDD.map(lambda (mid,mname,gen):movie(int(mid),mname)).toDF()

notRatedDF = (movieDF.filter(~movieDF["id"].isin(userRatedMovie))
                    .withColumnRenamed("id","product"))

notRatedMovie = notRatedDF.withColumn("user",lit(userId)).select("user","product")

predictedDF = model.predictAll(notRatedMovie.rdd).toDF()
  
joinDF = (predictedDF.join(movieDF, movieDF["id"]==predictedDF["product"],"inner")
                        .drop(predictedDF["product"]).drop(movieDF["id"])
                        .drop(predictedDF["user"]))

joinDF = joinDF.sort("rating",ascending=False).drop(joinDF["rating"])

rcmndMovies = joinDF.rdd.map(lambda movie: movie[0]).take(10)

rcmndMovies = sc.parallelize(rcmndMovies).coalesce(1)

rcmndMovies.saveAsTextFile(output)
