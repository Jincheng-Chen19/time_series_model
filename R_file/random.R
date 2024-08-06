library(tidyr)
library(dplyr)
library(ggplot2)
library(lmerTest)
library(purrr)

# Set to target directory
setwd('E:/work/seq_env/lucky/random/')
random <- read.delim('./data/random.csv', sep = '\t')
fil_random <- random %>%
  group_by(mask) %>%
  reframe(mask = unique(mask), num = n()) %>%
  filter(num >= 10) 
random <- random[which(random$mask %in% fil_random$mask),]
result <- random %>%
  group_by(mask) %>%
  filter(t == max(random$t))
random <- random[which(random$mask %in% result$mask),]
input <- random %>%
  group_by(mask) %>%
  filter(t != max(random$t))
LSTM <- input[,c('mask','mean_level','t')] %>%
  spread(key = t, value = mean_level)
write.csv(LSTM, './data/LSTM_input.csv', quote = FALSE, row.names = FALSE)
write.csv(result, './data/LSTM_output.csv', quote = FALSE, row.names = FALSE)
input_new <- unique(input$mask) %>%
  map(~{
    print(.)
    mask= .
    input0 = input %>% filter(mask == !! mask)
    sum_MI= sum(input0$mean_level)
    print(sum_MI)
    mean_MI = mean(input0$mean_level)
    print(mean_MI)
    sd_MI = sd(input0$mean_level)
    print(sd_MI)
    fc = input0[which(input0$t == min(input0$t)),'mean_level']/
      input0[which(input0$t == max(input0$t)),'mean_level']
    print(fc[[1]])
    mean_deri = mean(abs(diff(input0$mean_level)))
    print(mean_deri)
    lm1=lm(mean_level~t,input0)
    print(summary(lm1))
    slope=lm1$coefficients[[2]]
    y_inter = lm1$coefficients[[1]]
    print(slope)
    print(y_inter)
    integral = (slope*(min(input0$t) + max(input0$t)) + 2*y_inter)/(max(input0$t) - min(input0$t)) 
    input_new = c(mask, sum_MI, mean_MI, mean_deri,fc[[1]], slope, integral)
    return(input_new)
  }) %>% as.data.frame() %>% t() %>% data.frame()
colnames(input_new) = c('mask','sum_MI','mean_MI', 'mean_deri','foldchange','slope','integral')
row.names(input_new) = c(1:nrow(input_new))


data <- data.frame(cbind(input_new, result$mean_level))
colnames(data)[ncol(data)] <- c('result')
