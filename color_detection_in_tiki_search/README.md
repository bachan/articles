# Color Detection in Tiki Search

Queries that mention color in Tiki Search together with usage of the corresponding "color" filter accomodate to around 2% of the total search volume in Tiki. We want to address that volume in a relevant way for our customers, so that if specific color was requested, we would only show products of that color in our results.

While the statement above sounds simple, it is in fact a complex problem that requires a set of multiple approaches combined to solve it. Here's what we have at our disposal:

1. Sellers can create multiple variations of their product within the same product entity. For example, if they sell a phone case for Samsung Galaxy M31, they can create a variation for each specific design of the case to let customers choose the design they like on the Product Detail Page. One of such variations could as well be product color and in this case sellers are required to provide a freetext color tag for each of the variations.

<https://tiki.vn/op-lung-dien-thoai-samsung-galaxy-m31-01301-silicone-deo-hang-chinh-hang-p58006022.html?spid=58006038>

2. While the above is an option, sellers are not obligated to do such variations, they can also upload products of each color as separate entities and in that case they are not required to provide any color tags.

<https://tiki.vn/op-lung-dien-thoai-samsung-galaxy-m31-silicon-deo-0156-brown05-hang-chinh-hang-p82969564.html?spid=94275326>

3. Freetext color tags provided by sellers can range from something generic and easy to understand like "Xanh lá" to something that is much harder to process automatically (TODO add 3 examples).

4. Customers also may address colors in very different ways in their queries (TODO add 1 good and 3 bad examples).

Our task here is to come up with an approach that will help us to match product color data from our sellers with what our customers use to address those colors in their queries.

This article is going to tell about how did we split this big problem into smaller ones and how did we use machine learning to solve all of them.

## The Three Problems

Our ultimate goal with the seller side part is to be able to detect the color for 100% of our catalog with reaonsable accuracy. The only information we have that can tell us about colors is color variation tags set by sellers, which we supposedly trust; and also full set of product images, that sellers upload to represent their selection. So we naturally think about 2 separate problems here:

Problem 1: How do we detect color for each uploaded product image?

Problem 2: How do we use product image colors to make a decision about the actual product's color?

From customer side we have their query texts, so here comes the problem number 3:

Problem 3: How do we tell if a query means looking for products of a specific color and what is that specific color (if any)?

With that being said, let's jump to our solutions one by one.

## Product Images: Naive Approach

tell about grabcut here

## Product Images: Convolutional Neural Network Approach

tell about CNN here

## Product Colors: Merging Images Together (Not Easy)

tell about collecting all detected images into final product color here (and all related problems)

## Queries: ...

tell how we detect color related queries

## Conclusion

...

## References

...

## We're Hiring!

...

