##linear_models.rb
class StochasticGradientDescent
  attr_reader :weights
  attr_reader :objective
  
  def initialize obj, w_0, lr = 0.01
    @objective = obj
    @weights = w_0
    @n = 1.0
    @lr = lr
  end
  
  def update x
    grad = @objective.grad(x, @weights)
    learning_rate = @lr / Math.sqrt(@n)
    @n += 1
    
    for key in @weights.keys
      @weights[key] -= learning_rate * grad[key]
    end
    @objective.adjust(@weights)
  end
end

# LR2
class LogisticRegressionLearner
  attr_reader :parameters
  attr_reader :weights  
  include Learner  

  def initialize regularization: 0.0, learning_rate: 0.01, batch_size: 20, epochs: 1
    @parameters = {"regularization" => regularization, 
      "learning_rate" => learning_rate, 
      "epochs" => epochs, "batch_size" => batch_size, "n" => 1.0}
    @weights = Hash.new {|h,k| h[k] = 0.0}
  end
  
  def predict example
    x = example["features"]    
    1.0 / (1 + Math.exp(-dot(@weights, x)))
  end
  
  def adjust w
    w.each_key {|k| w[k] = 0.0 if w[k].nan? or w[k].infinite?}
    w.each_key {|k| w[k] = 0.0 if w[k].abs > 1e5 }
  end
  
  def func data, w # loss fuction
    n = data.size
    
    ll = 0
    for i in 0..(n - 1)
      z = 0
      for key in data[i]["features"].keys
        z += data[i]["features"][key] * w[key]
      end
      #puts [i, z]
      ll += Math.log(1 + Math.exp(-data[i]["label"] * z))
    end
    ll = ll / n + 0.5 * @parameters["regularization"] * dot(w, w)
  end
  
  def grad data
    g = Hash.new {|h,k| h[k] = 0.0}
    n = data.size
    
    for i in 0..(n - 1)
      #puts data[i]["features"].keys
      for key in data[i]["features"].keys
        xij = data[i]["features"][key]
        if data[i]["label"] == 1.0
          y = 1.0
        else
          y = 0.0
        end
        g[key] -= (y - predict(data[i])) * xij / n.to_f
      
        if i == n - 1
          g[key] += @parameters["regularization"] * @weights[key]
        end
      end
    end
    return g
  end
  
  def update x # update weights
    grad = grad(x)
    learning_rate = @parameters["learning_rate"] / Math.sqrt(@parameters["n"])
    @parameters["n"] += 1
    
    for key in @weights.keys
      @weights[key] -= learning_rate * grad[key]
    end
    adjust(@weights)
  end
  
  def score_binary_classification_model data
    scores = Array.new
    for i in 0..(data.size - 1)
      se = predict(data[i])
      scores.append([se, data[i]["label"]])
    end
    return scores
  end

  def train dataset
    n_batch_epoche = (dataset["data"].size / @parameters["batch_size"].to_f).ceil
    for e in 0..(@parameters["epochs"] - 1)
      for i in 0..(n_batch_epoche - 1) 
        batch_of_examples = dataset["data"][i * @parameters["batch_size"], @parameters["batch_size"]]
        update(batch_of_examples)
      end
    end
  end

  def evaluate dataset
    
      scores = score_binary_classification_model(dataset["data"])
  end
  
end