#' Compare performance between different models
#'
#' @param model_list a named list of different inference sessions you would
#' like to compare
#' @param data a dataframe to use in the testing. has to conform with the
#' format of the 'test_data' dataframe included in the package. If no data is
#' provided, the built-in testing dataset will be used.
#' @param n the size of the dataset used for evaluation.
#'
#' @export
race_models <- function(model_list, data = NULL, n=1000) {

    if (is.null(data)) {
        test_data <- test_data
    } else {
        test_data <- data
    }

    keys <- unique(test_data$key)
    sampled_keys <- sample(keys, n)
    data <- test_data %>%
        filter(key %in% sampled_keys) %>%
        mutate(idx = row_number())


    map_dfr(model_list, evaluate_one_model, data = data,.id = "model_name")
}

evaluate_one_model <- function(model, data){
    t1 <- Sys.time()
    embeddings <- generate_embeddings(model, pull(data, name))
    t2 <- Sys.time()

    seconds_elapsed <- difftime(t2, t1, units = "secs")

     match_df <- expand_grid(
            data %>% rename_with(~paste0("x_", .x)),
            data %>% rename_with(~paste0("y_", .x))
            ) %>%
        mutate(
               dist = map2_dbl(x_idx, y_idx, ~ sum((embeddings[.x, ] - embeddings[.y,])^2)),
               match = x_key == y_key
           )

     ks_test_out <- suppressWarnings(ks.test(match_df$dist[match_df$match], match_df$dist[!match_df$match]))

    plot <- match_df %>%
        ggplot(aes(x=dist, fill = match)) +
        geom_density()

     return(
            tibble(
                ks_statistic = ks_test_out$statistic,
                mean_match_dist = mean(match_df$dist[match_df$match]),
                mean_non_match_dist = mean(match_df$dist[!match_df$match]),
                seconds_elapsed = seconds_elapsed,
                density_plot = list(plot),
                n = nrow(data)
            ))
}



