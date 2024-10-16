# library(tidyverse)

# rextendr::document()
# devtools::load_all()

# Sys.setenv(ORT_DYLIB_PATH = "/nix/store/ifqds33j9scpwyiks89g86rqb1cqzhvc-onnxruntime-1.18.1/lib/libonnxruntime.so")

simple_by_validate <- function(a, b, by) {
  # first pass to handle dplyr::join_by() call
  if (inherits(by, "dplyr_join_by")) {
    if (any(by$condition != "==")) {
      stop("Inequality joins are not supported.")
    }
    new_by <- by$y
    names(new_by) <- by$x
    by <- new_by
  }

  if (is.null(by)) {
    by_a <- intersect(names(a), names(b))
    by_b <- intersect(names(a), names(b))
    stopifnot("Can't Determine Column to Match on" = length(by_a) == 1)
    message(paste0("Joining by '", by_a, "'\n"))
  } else {
    if (!is.null(names(by))) {
      by_a <- names(by)
      by_b <- unname(by)
    } else {
      by_a <- by
      by_b <- by
    }

    stopifnot(by_a %in% names(a))
    stopifnot(by_b %in% names(b))
  }
  return(list(
    by_a,
    by_b
  ))
}

neural_join <- function(model,a,b,by, radius = .1, exhaustive=FALSE, mode = "full") {
    by <- simple_by_validate(a, b, by)
    by_a <- by[[1]]
    by_b <- by[[2]]

    a_vec <- dplyr::pull(a, by_a)
    b_vec <- dplyr::pull(b, by_b)

    a_embeds <- generate_embeddings(model, a_vec)
    b_embeds <- generate_embeddings(model, b_vec)


    if (exhaustive) {
        match_table <- grid.expand(seq(length(a_vec)), seq(length(b_vec)))
    } else {
        match_table <- hnsw_join(a_embeds, b_embeds)
    }

    dist <- multi_cos_distance(a_embeds, b_embeds, match_table)

    within_dist <- dist < max_dist

    dist <- dist[within_dist]
    match_table <- match_table[within_dist, ]



      # Rename Columns in Both Tables
      names_in_both <- intersect(names(a), names(b))
      names(a)[names(a) %in% names_in_both] <- paste0(names(a)[names(a) %in% names_in_both], ".x")
      names(b)[names(b) %in% names_in_both] <- paste0(names(b)[names(b) %in% names_in_both], ".y")

      matches <- dplyr::bind_cols(a[match_table[, 1], ], b[match_table[, 2], ])
      matches$dist <- dist



  # No need to look for rows that don't match
  if (mode == "inner") {
    return(matches)
  }

  switch(mode,
    "left" = {
      not_matched_a <- collapse::`%!iin%`(seq(nrow(a)), match_table[, 1])
      matches <- dplyr::bind_rows(matches, a[not_matched_a, ])
    },
    "right" = {
      not_matched_b <- collapse::`%!iin%`(seq(nrow(b)), match_table[, 2])
      matches <- dplyr::bind_rows(matches, b[not_matched_b, ])
    },
    "full" = {
      not_matched_a <- collapse::`%!iin%`(seq(nrow(a)), match_table[, 1])
      not_matched_b <- collapse::`%!iin%`(seq(nrow(b)), match_table[, 2])
      matches <- dplyr::bind_rows(matches, a[not_matched_a, ], b[not_matched_b, ])
    },
    "anti" = {
      not_matched_a <- collapse::`%!iin%`(seq(nrow(a)), match_table[, 1])
      not_matched_b <- collapse::`%!iin%`(seq(nrow(b)), match_table[, 2])
      matches <- dplyr::bind_rows(a[not_matched_a, ], b[not_matched_b, ])
    }
  )

  matches
}
