##decision_trees.rb

#### KEEP THIS AT THE TOP OF YOUR FILE ####

SEED = 'eifjcchdivlbcbflbgblfgukbtkhvejvtkevfbtetjnl'.to_i(26)
module DecisionTreeHelper
  def to_s
    JSON.pretty_generate(summarize_node(@root))
  end
  
  def summarize_node node
    summary = {
      leaf: node.is_leaf?    
    }
    if node.is_leaf?
      summary[:class_distribution] = node.node_class_distribution
    else
      summary[:split] = node.split
      summary[:children] = node.children
        .sort_by{|kv| kv.first}
        .map do |kv|
          path, child = kv      
          [path, summarize_node(child)]
        end.to_h
    end

    return summary
  end
end


### ADD YOUR CODE AFTER THIS LINE ###
