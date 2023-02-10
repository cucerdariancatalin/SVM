import org.apache.commons.math3.linear.ArrayRealVector
import org.apache.commons.math3.linear.RealVector
import java.util.*

// Class to implement the Support Vector Machine (SVM) Algorithm
class SVM {
    private var weights: RealVector? = null
    private var bias: Double = 0.0

    // Function to train the SVM model using the training data
    fun train(trainingData: List<Pair<RealVector, Double>>, learningRate: Double, iterations: Int) {
        // Initialize the weights and bias values to zero
        weights = ArrayRealVector(trainingData[0].first.dimension)
        bias = 0.0

        // Loop through the specified number of iterations
        for (i in 0 until iterations) {
            // Loop through the training data
            for (pair in trainingData) {
                // Calculate the dot product of the weights and the input features
                val z = weights!!.dotProduct(pair.first) + bias

                // Calculate the predicted output using the activation function
                val prediction = if (z >= 0) 1.0 else -1.0

                // Calculate the error
                val error = pair.second - prediction

                // Update the weights and bias using the gradient descent algorithm
                weights = weights!!.add(pair.first.mapMultiply(error * learningRate))
                bias += error * learningRate
            }
        }
    }

    // Function to make predictions using the trained SVM model
    fun predict(input: RealVector): Double {
        // Calculate the dot product of the weights and the input features
        val z = weights!!.dotProduct(input) + bias

        // Return the predicted output using the activation function
        return if (z >= 0) 1.0 else -1.0
    }
}

// Main function to run the SVM algorithm
fun main() {
    val svm = SVM()

    // Create a list of training data
    val trainingData = listOf(
        Pair(ArrayRealVector(doubleArrayOf(1.0, 1.0)), 1.0),
        Pair(ArrayRealVector(doubleArrayOf(-1.0, -1.0)), -1.0),
        Pair(ArrayRealVector(doubleArrayOf(1.0, -1.0)), 1.0),
        Pair(ArrayRealVector(doubleArrayOf(-1.0, 1.0)), -1.0)
    )

    // Train the SVM model using the training data
    svm.train(trainingData, 0.1, 1000)

    // Make predictions using the trained SVM model
    val input = ArrayRealVector(doubleArrayOf(1.0, 1.0))
    println("Prediction for input [1.0, 1.0]: " + svm.predict(input))

    input.set(0, -1.0)
    input.set(1, -1.0)
    println("Prediction for input [-1.0, -1.0]: " + svm.predict(input))
}
