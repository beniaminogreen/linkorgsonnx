library(microbenchmark)

rextendr::document()
devtools::load_all()

data <- read.csv("../python_linkorgs_models/data/training_data.csv")

input <- character_tokenize(data[1:20,1], 40)

session <- ORTSession$new_from_path(4, "/home/beniamino/programming/python_linkorgs_models/lstm_embedding.onnx")
session <- ORTSession$new_from_path(4, "/home/beniamino/programming/Rort/gpt2.onnx")
session <- ORTSession$new_from_url(4, "https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/gpt2.onnx")

# strings <- c(
#     "The quick brown fox jumped over the lazy dog",
#     "The lazy dog was jumped over by the quick brown fox",
#     "The lazy dog was sleeping when the quick brown fox leaped over it",
#     "The quick brown fox leapt over the lazy dog",
#     "The lazy dog lay still as the quick brown fox jumped over",
#     "The quick brown fox vaulted over the lazy dog",
#     "I went to the store and bought apples and grapes",
#     "I went to the store and bought grapes and apples",
#     "I went to the store and bought batteries and a hammer",
#     "An instruction manual is a document that explains how to use a product or service",
#     "An instruction manual is meant to be a comprehensive resource for anything there is to know about a given product"
#              )

# embeddings <- reduce(map(strings, session$run_model), rbind)
# dimension_reduced_embeddings <- Rtsne(embeddings, perplexity=1)$Y
# plot_df <- tibble(
#     x = dimension_reduced_embeddings[,1],
#     y = dimension_reduced_embeddings[,2],
#     text = strings
#                   )

# plot_df %>%
#     ggplot(aes(x=x,y=y, label = text)) +
#     geom_point()  +
#     geom_text_repel() +
#     xlab("Dimension one") +
#     ylab("Dimension two") +
#     theme_bw(base_size = 20) +
#     scale_x_continuous(breaks = NULL) +
#     scale_y_continuous(breaks = NULL)

