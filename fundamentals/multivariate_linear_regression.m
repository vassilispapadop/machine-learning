clear all; close all; clc;
df = load('mv_regressionx.dat'); y = load('mv_regressiony.dat');
%combine independent with dependent variable
df = [df, y];
N = length(df);
