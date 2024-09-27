library(tidyverse)
rextendr::document()
devtools::load_all()

# snowflake_session <- inference_session_from_file(
#                                        "_model_stuff/snowflake-arctic-embed-m/model_quantized.onnx",
#                                        "_model_stuff/snowflake-arctic-embed-m/tokenizer.json",
#                                        num_threads = 12
#         )
# gte_tiny_session <- inference_session_from_file(
#                                        "_model_stuff/gte-tiny/model_quantized.onnx",
#                                        "_model_stuff/gte-tiny/tokenizer.json",
#                                        num_threads = 12
#         )
all_MiniLM_L6_v2_session<- inference_session_from_file(
                                       "_model_stuff/all-MiniLM-L6-v2/model.onnx",
                                       "_model_stuff/all-MiniLM-L6-v2/tokenizer.json",
                                       num_threads = 12
        )
fine_tuned <- inference_session_from_file(
                                       "_model_stuff/fine_tuned_model/test_output/model.onnx",
                                       "_model_stuff/fine_tuned_model/test_output/tokenizer.json",
                                       num_threads = 12
        )

# GIST_quantized_session <- inference_session_from_file(
#                                        "_model_stuff/GIST/model_quantized.onnx",
#                                        "_model_stuff/GIST/tokenizer.json",
#                                        num_threads = 12
#         )
# GIST_session <- inference_session_from_file(
#                                        "_model_stuff/GIST/model.onnx",
#                                        "_model_stuff/GIST/tokenizer.json",
#                                        num_threads = 12
#         )


test_output_df <- race_models(
            list(
                 # snowflake = snowflake_session,
                 # gte_tiny = gte_tiny_session,
                 all_MiniLM_L6_v2 = all_MiniLM_L6_v2_session,
                 fine_tuned = fine_tuned
                 # GIST_quantized = GIST_quantized_session,
                 # GIST = GIST_session
            )
            , n=10)

# test_output_df$seconds_elapsed


# data <- read_csv("../python_linkorgs_models/data/training_data.csv") %>%
#     arrange(key) %>%
#     mutate(
#            idx = row_number()
#            )

# X <- run_model(gte_tiny_session, pull(head(data,5000), name))


# # out_df <- expand_grid(
# #             data %>% rename_with(~paste0("x_", .x)),
# #             data %>% rename_with(~paste0("y_", .x))
# #             )
# # out_df <- out_df %>%
# #     mutate(
# #            dist = map2_dbl(x_idx, y_idx, ~ sum((X[.x, ] - X[.y,])^2)),
# #            match = x_key == y_key
# #            )

# # out_df %>%
# #     ggplot(aes(x=dist, fill = match)) +
# #     geom_density()




















































# # # # # out_df %>%
# # # # #     filter(match = TRUE, dist > .15) %>%
# # # # #     select(x_name, y_name)

# # # # # session <- ORTSession$new_from_path(4, "/home/beniamino/programming/Rort/gpt2.onnx")
# # # # # session <- ORTSession$new_from_url(4, "https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/gpt2.onnx")

# # # # # strings <- c(
# # # # #     "The quick brown fox jumped over the lazy dog",
# # # # #     "The lazy dog was jumped over by the quick brown fox",
# # # # #     "The lazy dog was sleeping when the quick brown fox leaped over it",
# # # # #     "The quick brown fox leapt over the lazy dog",
# # # # #     "The lazy dog lay still as the quick brown fox jumped over",
# # # # #     "The quick brown fox vaulted over the lazy dog",
# # # # #     "I went to the store and bought apples and grapes",
# # # # #     "I went to the store and bought grapes and apples",
# # # # #     "I went to the store and bought batteries and a hammer",
# # # # #     "An instruction manual is a document that explains how to use a product or service",
# # # # #     "An instruction manual is meant to be a comprehensive resource for anything there is to know about a given product"
# # # # #              )

# # # # # embeddings <- reduce(map(strings, session$run_model), rbind)
# # # # # dimension_reduced_embeddings <- Rtsne(embeddings, perplexity=1)$Y
# # # # # plot_df <- tibble(
# # # # #     x = dimension_reduced_embeddings[,1],
# # # # #     y = dimension_reduced_embeddings[,2],
# # # # #     text = strings
# # # # #                   )

# # # # # plot_df %>%
# # # # #     ggplot(aes(x=x,y=y, label = text)) +
# # # # #     geom_point()  +
# # # # #     geom_text_repel() +
# # # # #     xlab("Dimension one") +
# # # # #     ylab("Dimension two") +
# # # # #     theme_bw(base_size = 20) +
# # # # #     scale_x_continuous(breaks = NULL) +
# # # # #     scale_y_continuous(breaks = NULL)

