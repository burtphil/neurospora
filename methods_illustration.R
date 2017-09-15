### method script illustration of period amplitude and interpolation and phase?

setwd("/home/burt/neurospora/figures")

require(ggplot2)
require(ggthemes)


poly <- function(x) {
  -0.5*x^2+2.5*x-1
}

sine <- function(x){
  sin(pi*x)
}

const <- function(x){
  0
}

### define a theme for plotting
size_axis <- element_text(face="bold", size = 30)
size_ticks <- element_text(face="bold", size = 30, color = "black")
size_line <- element_line(color = "black", size = 2)

theme_thesis <- theme_few() +
  theme(axis.title = size_axis,
        axis.title.y = element_text(margin = margin(0, 0.8, 0, 0, "cm")),
        axis.title.x = element_text(margin = margin(0.5, 0, 0, 0, "cm")),
        axis.text = size_ticks,
        axis.ticks = size_line,
        axis.ticks.length=unit(2,"mm"),
        panel.border = element_rect(size = 2, color = "black")
  )

x = c(1,3,4)
y = c(1,2,1)
xm = 2.5
ym = 2.125
df2 = data.frame(xm,ym)
df = data.frame(x,y)

p <- ggplot(data.frame(x = c(0, 6)), aes(x = x)) +
  stat_function(fun = poly, size = 1.2)+
  geom_point(data=df, aes(x=x,y=y), color = "red", size = 4)+
  geom_point(data=df2, aes(x=xm,y=ym), color = "blue", size = 4)+
  scale_shape_discrete(solid=F) +
  coord_cartesian(xlim = c(0,5), ylim = c(0,3), expand = TRUE)+
  theme_thesis

print (p)

ggsave("interpol.pdf", dpi = 1200)

a = c(0.5,2.5)
b = c(1,1)
df3 = data.frame(a,b)

f <- ggplot(data.frame(x = c(-1, 5)), aes(x))

f + stat_function(fun = sine, size =1.2)+
  coord_cartesian(ylim = c(-1.5,1.5),xlim = c(0,3.2))+
  geom_hline(yintercept = 0, size =1.2)+
  geom_point(data = df3, aes(x=a,y=b), color = "red", size = 4)+
  theme_thesis

ggsave("amp_per.pdf", dpi = 1200)

