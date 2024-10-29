#include <stdio.h>
#include <stdlib.h>

double piRectangles(int intervals){
    double width = 1.0/intervals;
    double sum = 0.0, x;
    for(int i = 0; i<intervals; i++){
        x = (i + 0.5)*width;
        sum += 4.0/(1.0 + x*x);
    }
    return sum*width;
}

int main(int argc, char* argv[]){
    if(argc!=2){    //El primer argumento siempre es el nombre del programa
        printf("Uso: ./prog esfuerzo\n");
    }else{
        int steps = atoi(argv[1]);
        if(steps<=0){
            printf("El numero de iteraciones debe ser un entero positivo!\n");
        }else{
            double pi = piRectangles(steps);
            printf("Valor de PI aproximado [%d intervalos] = \t%lf\n", steps, pi);
        }
    }
    return 0;
}
