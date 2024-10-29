#ifndef __PGM__
#define __PGM__

void ** GetMem2D(int rows, int columns, int sizeofTipo);
void Free2D(void ** h);
unsigned char** pgmread(char* filename, int* rows, int* columns);

// Ver https://netpbm.sourceforge.net/doc/pgm.html para mas detalles

#endif /*__PGM__*/
