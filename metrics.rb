### metrics.rb
# mean
def mean x
  sum = 0.0
  x.each{|a| sum += a}
  sum / x.size.to_f
end

# stdev
def stdev x
  m = mean(x)
  variance = 0.0
  x.each{|v| variance += (v - m) ** 2}
  Math.sqrt(variance / (x.size.to_f - 1))
end

def dot x, w
  prod = 0
  for key in x.keys
    if w.has_key?(key)
      prod += x[key] * w[key]
    end
  end
  prod
end

def norm w
  Math.sqrt(norm)
end

#  class AUCMetric
def num_positives scores
  num = 0
  for i in 0..(scores.size - 1)
    if scores[i][1] == 1.0
      num += 1
    end
  end
  num
end

def num_negatives scores
  num_pos = 0
  for i in 0..(scores.size - 1)
    if scores[i][1] == 1.0
      num_pos += 1
    end
  end
  scores.size - num_pos
end

def confusion_matrix(scores, t)
  matrix = Hash.new {|h,predicted_class| h[predicted_class] = Hash.new {|h,true_class| h[true_class] = 0.0}}
  for i in 0..(scores.size - 1)
    if scores[i][0] >= t
      if scores[i][1] == 1.0
        matrix["P"]["P"] += 1
      else
        matrix["P"]["N"] += 1
      end
    else
      if scores[i][1] == 1.0
        matrix["N"]["P"] += 1
      else
        matrix["N"]["N"] += 1
      end
    end
  end
  return matrix
end

def false_positive_rate(matrix, total_pos, total_neg)
  matrix["P"]["N"] / total_neg.to_f
end

def true_positive_rate(matrix, total_pos, total_neg)
  matrix["P"]["P"] / total_pos.to_f
end

module Metric
  def apply scores
  end
end

class AUCMetric 
  include Metric
  
  def roc_curve(scores)
    fp_rates = [0.0]
    tp_rates = [0.0]
    auc = 0.0
    
    scores.sort_by!{|a| -a[0]}
  
    np = num_positives(scores)
    nn = num_negatives(scores)
    ni_p = 0
    ni_n = 0
    for i in 0..(scores.size - 1)
      if scores[i][1] == 1.0
        ni_p += 1
      else
        ni_n += 1
      end
      
      f = ni_n / nn.to_f
      t = ni_p / np.to_f
      trapezoids = 0.5 * (f - fp_rates[i]) * (t + tp_rates[i])
      auc += trapezoids 
      
      fp_rates.append(f)
      tp_rates.append(t)
    end
    
    return [fp_rates, tp_rates, auc]
  end
  
  def apply scores
    fp, tp, auc = roc_curve scores
    auc
  end
end

# cross_validate
def cross_validate dataset, folds, &block
  examples = dataset["data"]
  fold_size = examples.size / folds
  folds.times do |fold|
    ##CV training examples
    train_data = dataset.clone
    train_data["data"] = train_data["data"][0, fold * fold_size] + train_data["data"][((fold + 1) * fold_size)..-1]
    
     ##CV testing examples
    test_data = dataset.clone
    test_data["data"] = test_data["data"][fold * fold_size, fold_size]             

    ## Call the callback like this:
    yield train_data, test_data, fold
  end
end


#### KEEP THIS AT THE TOP OF YOUR FILE ####
def plot_roc_curve fp, tp, auc
  plot = Daru::DataFrame.new({x: fp, y: tp}).plot(type: :line, x: :x, y: :y) do |plot, diagram|
    plot.x_label "False Positive Rate"
    plot.y_label "True Positive Rate"
    diagram.title("AUC: %.4f" % auc)
    plot.legend(true)
  end
end  

def cross_validation_model_performance dataset, folds, learners, metric    
  learners.map do |learner|
    tr_metrics = []
    te_metrics = []
    puts "#{folds}-fold CV: #{learner.class.name}, parameters: #{learner.parameters}"
    cross_validate dataset, folds do |train_dataset, test_dataset|
      learner.train train_dataset
      train_scores = learner.evaluate train_dataset
      test_scores = learner.evaluate test_dataset      
      tr_metrics << metric.apply(train_scores)
      te_metrics << metric.apply(test_scores)
    end
      
    #Train on full training set
    learner.train dataset
    learner_name = learner.name
    puts mean(te_metrics)
    {
      "learner" => learner_name, "trained_model" => learner, "parameters" => learner.parameters, "folds" => folds,
      "mean_train_metric" => mean(tr_metrics), "stdev_train_metric" => stdev(tr_metrics),
      "mean_test_metric" => mean(te_metrics), "stdev_test_metric" => stdev(te_metrics),
    }
  end
end

def best_performance_by_learner stats  
  stats.group_by {|s| s["learner"]}.map do |g_s|
    learner, learner_stats = g_s
    best_parameters = learner_stats.max_by {|l| l["mean_test_metric"]}    
    [learner, best_parameters]
  end.to_h
end

def parameter_search learners, dataset, folds = 5
  metric = AUCMetric.new  
  stats = cross_validation_model_performance dataset, folds, learners, metric
  best_by_learner = best_performance_by_learner stats  
    summary = Hash.new
    best_by_learner.each_key do |k|
        summary[k] = best_by_learner[k].clone
        summary[k].delete "trained_model"
    end
  puts JSON.pretty_generate(summary)

  assert_equal learners.size, stats.size
  assert_true(stats.all? {|s| a = s["mean_train_metric"]; a >= 0.0 and a <= 1.0}, "0 <= Train AUC <= 1")
  assert_true(stats.all? {|s| a = s["mean_test_metric"]; a >= 0.0 and a <= 1.0}, "0 <= Train AUC <= 1")
  
  stats.map! {|s| t = s.clone; t.delete "trained_model"; t}
  df = Daru::DataFrame.new(stats) 
    
  return [df, best_by_learner]
end

### ADD YOUR CODE AFTER THIS LINE ###