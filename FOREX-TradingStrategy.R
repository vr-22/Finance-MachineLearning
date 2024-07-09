require(quantmod)
require(tseries)
require(rugarch)
eur_to_usd<-read.csv("C:\\Users\\Ritwij Verma\\OneDrive\\Documents\\algorithmic_trading\\TimeSeries\\TradingStrategy\\EURUSD.csv")
dates<-as.Date(as.character(eur_to_usd[,1]),format="%d/%m/%y",header=T)
eur_to_usd[,1]=dates
plot(eur_to_usd$C,type='l')
returns<-diff(log(eur_to_usd$C))
returns
window.length<-100
forecasts.length<-(length(returns)-window.length)
forecasts<-vector(mode = "numeric",length = forecasts.length)
directions<-vector(mode ="numeric",length = forecasts.length)
#making prediction for next trading day  
for(i in 0:forecasts.length)
{ 
  #creating a rolling window
  roll.returns=returns[(i+1):(window.length+i)]
  final.aic<-Inf
  final.order<-c(0,0,0)
  final<-NULL
  for(p in 1:4)for(q in 1:4)
  {
    model<-tryCatch(arima(roll.returns,order=c(p,0,q)),error= function(err)FALSE,warning=function(err)FALSE)
    #FALSE would mean that the model was unable to fit,ex: we nee more iterations
    if(!is.logical(model))
    {
      if(final.aic<AIC(model))
      {
        final.aic<-AIC(model)
        final.order<-c(p,0,q)
        final<-model
      }
    }
  }
      #we will be making a GARCH(1,1) model
      spec=ugarchspec(variance.model<-list(garchOrder=c(1,1)),
                      mean.model <- list(armaOrder<-c(final.order[1],final.order[3]),include.mean=T),
                      distribution.model="sged")#allows for a wide choice in uni-variate GARCH models, distributions, and mean equation modelling
      #SEGD=Skew Error Generalized Distribution
      #checking for exceptions in the model,fit=false if any
      fit=tryCatch(ugarchfit(spec,roll.returns,solver = "hybrid"), error = function(e) e, warning = function(w) w)
      #hybrid solver, is basically for ensure best possible fitting using various fitting algorithms available
      #making prediction for the next trading day if there is no error
      if(is(fit,"warning"))
      {
        forecasts[i+1]<-0
      }
      else
      {
        #predicting tomorrow's returns
        next.day.forecast=ugarchforecast(fit,n.ahead=1)
        #x will be storing the predicted value for the next day
        x1=next.day.forecast@forecast$seriesFor
        #we also need direction for +return and negative returns
        directions[i+1]<-ifelse(x1[1]>0,+1,-1)
        forecasts[i+1]<-x1[1]
      }
}
forecasts
forecast.ts<-xts(forecasts,dates[(window.length):(length(returns))])
strategy.forecast<-Lag(forecast.ts,1)
strategy.direction<-ifelse(strategy.forecast>0,1,ifelse(strategy.forecast<0,-1,0))
strategy.direction.returns<-strategy.direction*returns[(window.length):(length(returns))]#XTS=Extended Time Series
strategy.direction.returns[1] <- 0 # as the first value returned Na(Invalid)
#to build returns we need take cumulative sums of daily returns
strategy.curve.data<-cumsum(strategy.direction.returns)
#comparing it with returns of long term investing
long.-term.ts<-xts(returns[(window.length):length(returns)],dates[(window.length):length(returns)])
long.term.curve<-cumsum(long.term.ts)
#clubbing them together
both.curves<-cbind(strategy.curve.data,long.term.curve)
#renaming for better organisation
names(both.curves) <- c("Strategy Returns", "Long Term Investing Returns")
#plot ARIMA+GARCH strategy as well as the long term investing method
plot(x = both.curves[,"Strategy Returns"], xlab = "Time", ylab = "Cumulative Return",
     main = "Cumulative Returns", ylim = c(-0.25, 0.4), major.ticks= "quarters",
     minor.ticks = FALSE, col = "green")
lines(x = both.curves[,"Long Term Investing Returns"], col = "red")
strategy_colors <- c( "green", "red") 
legend(x = 'bottomleft', legend = c("ARIMA&GARCH", "Long Term Investing"),
       lty = 1, col = strategy_colors)






