## transformers.rb

#### KEEP THIS AT THE TOP OF YOUR FILE ####
class TransformingLearner
  include Learner
  attr_accessor :name
  def initialize transformer, learner
    @parameters = learner.parameters
    @parameters["learner"] = learner.parameters["name"] || learner.class.name
    @transformer = transformer
    @learner = learner
    @name = self.class.name
  end
  def train dataset
    @transformer.train dataset
    transformed_examples = @transformer.apply dataset["data"]
    train_dataset = dataset.clone
    train_dataset["data"] = transformed_examples
    @learner.train train_dataset
  end
  
  def predict example
    transformed_example = @transformer.apply [example]
    @learner.predict transformed_example.first
  end
  
  def evaluate dataset
    transformed_dataset = dataset.clone
    transformed_dataset["data"] = @transformer.apply dataset["data"]
    @learner.evaluate transformed_dataset
  end
end

class CopyingTransformingLearner
  include Learner
  attr_accessor :name
  def initialize transformer, learner
    @parameters = learner.parameters
    @parameters["learner"] = learner.parameters["name"] || learner.class.name
    @transformer = transformer
    @learner = learner
    @name = self.class.name
  end

  def clone_example example
    e = example.clone
    e["features"] = example["features"].clone
    return e
  end
    
  def clone_dataset dataset
    cloned_dataset = dataset.clone
    cloned_dataset["features"] = dataset["features"].clone
    cloned_dataset["data"] = dataset["data"].map {|e| clone_example(e)}
    return cloned_dataset
  end
    
  def train dataset
    @transformer.train clone_dataset(dataset)
      
    train_dataset = clone_dataset(dataset)
    transformed_examples = @transformer.apply train_dataset["data"]
    train_dataset["data"] = transformed_examples
    @learner.train train_dataset
  end
  
  def predict example
    transformed_example = @transformer.apply [clone_example(example)]
    @learner.predict transformed_example.first
  end
  
  def evaluate dataset
    transformed_dataset = clone_dataset dataset
    transformed_dataset["data"] = @transformer.apply dataset["data"]
    @learner.evaluate transformed_dataset
  end
end


### ADD YOUR CODE AFTER THIS ###
