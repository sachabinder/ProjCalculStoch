package BlackScholes.src;
import java.lang.Math;
import BlackScholes.lib.Gaussian;

public class App {

    class BlackScholes{
        private double r, sigma;
        private int T;
        private double K;
        private int mu_p, mu_c;


        BlackScholes(double r, double sigma, int T, double K, int mu_p, int mu_c){
            this.r = r;
            this.sigma = sigma;
            this.T = T;
            this.K = K;
            this.mu_p = mu_p;
            this.mu_c = mu_c;
        }

        double d(double x, double K){
            double d_num = (Math.log(x/K) + this.T * (this.sigma + Math.pow(this.sigma, 2) / 2));
            double d_denom = this.sigma * Math.sqrt(this.T);
            return d_num/d_denom;
        }

        double call(double x, int t){
            double dp = d(x, this.K);
            double dc = dp - this.sigma * Math.sqrt(this.T);
            return x * Gaussian.cdf(dp) - this.K * Math.exp(-this.r * this.T) * Gaussian.cdf(dc);
        }

        double put(double x, int t){
            return call(x, t) - x + this.K * Math.exp(-this.r * this.T);
        }

        double Q(double x, int t){
            return this.mu_p * put(x, this.T - t) + this.mu_c * call(x, this.T - t);
        }

        double Delta(double x, int t){
            double eps = 1/Math.pow(10, 6);
            return (Q(x + eps, t) - Q(x - eps, t))/ (2 * eps);
        }

    }

    public static void main(String[] args) throws Exception {
        System.out.println("Hello, World!");
    }
}
