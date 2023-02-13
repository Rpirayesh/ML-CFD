clear all
clc
close all
load('PerNNHL')
load('PerNNNeurons')
MinPerN=min(PerformanceAve,[],2);
MinPerL=min(PerformanceAveHL,[],2);
figure
plot(MinPerN)
xlabel('Batches')
ylabel('Performance')
grid
figure
plot(MinPerL)
xlabel('Batches')
ylabel('Performance')
grid

figure
plot(PerformanceAveHL(end,:))