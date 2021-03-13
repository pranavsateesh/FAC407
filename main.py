from fac407_prnvstsh_module import plot_the_closing, plot_the_adjclosing, plot_the_net_returns, plot_the_gross_returns, plot_the_logret, distribution_returns_normdist, get_mean_median_mode_logret, normal_test_ad, normal_test_sw, correlation
import os

def main():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        var = input(" Hello Choose a number for the result/analysis \
        1. Plot the closing price \
        2. Plot the adjusted closing price \
        3. Plot the net returns \
        4. Plot the gross returns \
        5. Plot the log returns \
        6. Plot the Distribution of Returns \
        7. Statiscal information about the log-returns \
        8. Normal distribution test Shapiro-Wilk \
        9. Normal distribution test Anderson-Darling \
        10. Correlation \
        11. EXIT ")
        if var == "1":
            plot_the_closing()
        elif var == "2":
            plot_the_adjclosing()
        elif var == "3":
            plot_the_net_returns()
        elif var == "4":
            plot_the_gross_returns()
        elif var == "5":
            plot_the_logret()
        elif var == "6":
            distribution_returns_normdist()
        elif var == "7":
            get_mean_median_mode_logret()
        elif var == "8":
            normal_test_sw()
        elif var == "9":
            normal_test_ad()
        elif var == "10":
            correlation()
        elif var == "11":
            exit()

if __name__ == "__main__":
    main()
